#!/usr/bin/env python3
from __future__ import annotations
"""
Curriculum-aware Token-UNet Trainer
==================================
- Trains TokenUNet on curriculum stages, advancing only when validation loss plateaus or training loss stagnates.
- Dynamically adjusts dropout, bottleneck, learning rate, batch size, and gradient clipping per stage.
- Robust to NaNs, OOMs, and resumes from checkpoint.
- Enhanced plotting and logging for deep debugging and analysis.
- Stores outputs in CurriculumTraining/stageX_name/ckpts, imgs, logs.
- 48kHz, 24kbps EnCodec enforced throughout.

Stage Advancement Rules:
- Advance when:
    - Validation loss plateaus (3-epoch moving average, abs(ema[-1] - ema[-6]) < 0.05 for 5 epochs, and only after min_epochs_per_stage)
    - Or training loss stagnates for 8+ epochs
- min_epochs_per_stage = max(10, n_train // 100)
- Apply higher dropout (0.15) in stage 4+
- Switch use_bottleneck=True in stage4_full_stronger
- Shrink LR by 25–50% each stage (start 1e-3, down to 5e-6 in later stages)
- OneCycleLR with pct_start from STAGES tuple
- AdamW (foreach=False, fused=False)
- Validation/test split (80/10/10), fixed seed (min 1 for val/test)
- Dynamic batch size: increase if memory allows
- Adaptive gradient clipping: adjust based on grad norm stats
- Minimal but critical prints for debugging, warnings, and stage transitions
- Save/restore all state for robust resume
- Save best model globally as CurriculumTraining/best_overall.pt
- Save per-stage logs as training_log.csv in stage_dir/logs/
- --until_stage CLI flag to stop at an intermediate stage
- (Future: attention memory optimization if needed)
"""

# Set CUDA memory allocator config before importing torch
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


import argparse
import csv
import gc
import math
import os
import random
import shutil
import time
import traceback
import warnings
from datetime import datetime
from pathlib import Path
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import OneCycleLR, SequentialLR, LinearLR, ConstantLR, CosineAnnealingLR
from torch.utils.data import DataLoader, random_split, Subset
import torchaudio
from tqdm import tqdm
from collections import defaultdict
import typing as tp

from token_dataset import TokenPairDataset, pad_collate
from token_unet import TokenUNet, PAD
from encodec import EncodecModel
from token_utils import compute_metrics, save_audio
from token_train_utils import (
    create_scheduler, handle_oom, handle_nan, compute_grad_norm_and_max,
    analyze_token_coverage, verify_codec_settings,
    TRAIN_PLATEAU_EPOCHS, GRAD_NORM_WARN
)
from token_plot_utils import (
    plot_curves, print_training_stats, print_stage_transitions,
    print_batch_stats, print_optimization_stats, print_model_stats
)
from token_data_utils import (
    create_loaders, get_epochs_per_stage, get_min_epochs_per_stage,
    verify_train_val_split, flatten_logits_targets, print_dataset_stats
)

# Constants
BATCH_START = 4
BATCH_MAX = 32
BATCH_INC_EPOCHS = 5
VAL_FRAC, TEST_FRAC = 0.10, 0.10

# Curriculum stages: (name, dropout, use_bottleneck, max_lr, pct_start, n_train)
STAGES = [
    ("stage0_identity",       0.00, False, 5e-4, 0.05, 1),
    ("stage1_single",         0.00, True,  4e-5, 0.15, 4),
    ("stage1_single_stronger",0.00, True,  3e-5, 0.15, 4),
    ("stage2_double",         0.05, True,  2e-5, 0.10, 4),
    ("stage3_triple",         0.08, True,  1.5e-5, 0.08, 4),
    ("stage3_triple_stronger",0.10, True,  1e-5, 0.05, 4),
    ("stage4_full",           0.15, True,  8e-6, 0.02, 4),
    ("stage4_full_stronger",  0.15, True,  5e-6, 0.02, 4),
]

# Loss weights
SPECTRAL_WEIGHT = 0.1    # Weight for STFT magnitude loss
TIME_WEIGHT = 0.1        # Weight for waveform L1 loss
AUDIO_WEIGHT = 0.01      # Weight for mel-spectrogram loss
TOKEN_SPEC_WEIGHT = 0.05 # Weight for token-space spectral loss

# Stage-specific parameters
STAGE0_MIN_EPOCHS = 100
STAGE1_MIN_EPOCHS = 150
STAGE0_VAL_THRESHOLD = 0.5
STAGE1_VAL_THRESHOLD = 0.01
STAGE1_ABS_THRESHOLD = 0.5
STAGE1_MAX_PATIENCE = 12
MAX_PATIENCE = 6
TRAIN_PLATEAU_EPOCHS = 8
GRAD_NORM_WARN = 20
SEED = 42
CLIP_BASE = 0.3  # Reduced from 0.5
CLIP_MIN = 0.1
CLIP_MAX = 1.0  # Reduced from 2.0
CLIP_WINDOW = 5
WARMUP_CLIP_EPOCHS = 3

# Stage 0 specific settings
STAGE0_BATCH_SIZE = 16  # Much larger batch size for stage 0
STAGE0_GRAD_ACCUM = 1   # No gradient accumulation needed for stage 0

# Scheduler configuration
DIV_FACTOR_STAGE0 = 10
FINAL_DIV_FACTOR = 100  # Gentler decay

# --- quick test pass criteria ----------------------------------
TARGET_LOSS   = 6.0      # ~13% better than random
MAX_UPDATES   = 400      # 50 epochs over the 32-sample subset
PRINT_EVERY   = 20

# ──────────────── Module-level Transforms ────────────────
_HANN_WINDOW = torch.hann_window(2048)
_MEL_TRANSFORM = torchaudio.transforms.MelSpectrogram(
    sample_rate=48000,
    n_fft=2048,
    hop_length=512,
    n_mels=128
)

# Move transforms to GPU if available
if torch.cuda.is_available():
    _HANN_WINDOW = _HANN_WINDOW.cuda()
    _MEL_TRANSFORM = _MEL_TRANSFORM.cuda()

def spectral_convergence(S_pred: torch.Tensor, S_gt: torch.Tensor) -> torch.Tensor:
    """Compute spectral convergence loss."""
    return torch.norm(S_gt - S_pred, p='fro') / (torch.norm(S_gt, p='fro') + 1e-8)

def log_mse(S_pred: torch.Tensor, S_gt: torch.Tensor) -> torch.Tensor:
    """Compute log-MSE loss on spectrograms."""
    return F.mse_loss(torch.log1p(S_pred), torch.log1p(S_gt))

# ──────────────── Utility Functions ────────────────
def pretty(t): m,s=divmod(int(t),60); h,m=divmod(m,60); return f"{h}:{m:02d}:{s:02f}"
def set_seed(s): random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def print_gpu_stats(where):
    """Print GPU memory stats."""
    if torch.cuda.is_available():
        print(f"\nGPU stats at {where}:")
        print(f"Allocated: {torch.cuda.memory_allocated()/1e9:.1f}GB")
        print(f"Cached: {torch.cuda.memory_reserved()/1e9:.1f}GB")
        print(f"Max allocated: {torch.cuda.max_memory_allocated()/1e9:.1f}GB")

def print_training_stats(epoch, avg_train, avg_val, tnorm, lr, batch_size, clip_val, time_taken, stage_name):
    """Print training statistics for current epoch."""
    print(f"\nEpoch {epoch} ({stage_name})")
    print(f"Train loss: {avg_train:.4f}")
    print(f"Val loss: {avg_val:.4f}")
    print(f"Grad norm: {tnorm:.2f}")
    print(f"LR: {lr:.2e}")
    print(f"Batch size: {batch_size}")
    print(f"Clip value: {clip_val:.2f}")
    print(f"Time taken: {pretty(time_taken)}")

def print_stage_transition(stage_name, dropout, use_bottleneck, max_lr, pct_start, n_train):
    """Print stage transition information."""
    print(f"\nTransitioning to {stage_name}")
    print(f"Dropout: {dropout}")
    print(f"Use bottleneck: {use_bottleneck}")
    print(f"Max LR: {max_lr:.2e}")
    print(f"Warmup: {pct_start*100:.1f}%")
    print(f"Min epochs: {get_min_epochs_per_stage(n_train, stage_name)}")

def print_batch_stats(step, loss, grad_norm_tuple, clip_val, batch_size, grad_accum):
    """Print batch statistics."""
    tnorm, maxnorm = grad_norm_tuple  # Unpack the tuple
    print(f"Step {step}: loss={loss:.4f}, grad_norm={tnorm:.2f}, max_norm={maxnorm:.2f}, "
          f"batch={batch_size}, clip={clip_val:.2f}, accum={grad_accum}")

def print_optimization_stats(opt, sched):
    """Print optimization statistics."""
    print("\nOptimization stats:")
    print(f"LR: {opt.param_groups[0]['lr']:.2e}")
    print(f"Beta1: {opt.param_groups[0]['betas'][0]}")
    print(f"Beta2: {opt.param_groups[0]['betas'][1]}")
    print(f"Weight decay: {opt.param_groups[0]['weight_decay']}")
    if isinstance(sched, OneCycleLR):
        print(f"OneCycleLR: max_lr={sched.max_lrs[0]:.2e}, pct_start={sched.pct_start}")

def print_model_stats(net):
    """Print model statistics."""
    print("\nModel stats:")
    total_params = sum(p.numel() for p in net.parameters())
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")

def run_quick_test(n_samples: int, dev: torch.device):
    """Run a quick test pass to verify model and data pipeline."""
    print(f"\nRunning quick test with {n_samples} samples...")
    
    # Create small dataset
    ds = TokenPairDataset("experiments/curriculums", stages=["stage0_identity"])
    n_q = ds.n_q  # Get n_q before creating subset
    if len(ds) > n_samples:
        indices = torch.randperm(len(ds))[:n_samples]
        ds = Subset(ds, indices)
    
    # Create model
    net = TokenUNet(
        n_q=n_q,  # Use the saved n_q value
        dropout=0.0,
        use_bottleneck=False
    ).to(dev)
    
    # Create optimizer and scheduler
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4)
    sched = OneCycleLR(opt, max_lr=1e-3, total_steps=MAX_UPDATES)
    
    # Training loop
    net.train()
    total_loss = 0
    updates = 0
    
    for i, (x, y) in enumerate(tqdm(ds, desc="Quick test")):
        # Add batch dimension
        x = x.unsqueeze(0)  # [1, n_q, T]
        y = y.unsqueeze(0)  # [1, n_q, T]
        x, y = x.to(dev), y.to(dev)
        
        # Forward pass
        logits = net(x)  # [1, K, n_q, T]
        # Reshape for cross entropy
        logits = logits.permute(0, 2, 3, 1).reshape(-1, logits.size(1))  # [n_q*T, K]
        y = y.reshape(-1)  # [n_q*T]
        
        loss = F.cross_entropy(logits, y)
        
        # Backward pass
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        
        total_loss += loss.item()
        updates += 1
        
        if updates >= MAX_UPDATES:
            break
    
    avg_loss = total_loss / updates
    print(f"\nQuick test results:")
    print(f"Average loss: {avg_loss:.4f}")
    print(f"Target loss: {TARGET_LOSS:.4f}")
    print(f"Pass: {avg_loss < TARGET_LOSS}")
    
    return avg_loss < TARGET_LOSS

def save_validation_audio(net, val_loader, codec, epoch, output_dir, stage_name, dev):
    """Save validation audio samples."""
    try:
        # Create codec for audio generation
        codec = codec.to(dev)
        codec.set_target_bandwidth(24.0)
        
        # Process validation samples
        for i, batch in enumerate(val_loader):
            try:
                # Handle stage 0 differently
                if stage_name == "stage0_identity":
                    x, y, y_scales = batch
                else:
                    x, y, y_scales, Y_stft, mel_spec, stft_o, stft_m, mel_o, mel_m = batch
                
                # Move tensors to device
                x = x.to(dev)
                y = y.to(dev)
                y_scales = y_scales.to(dev)
                
                # Get predictions
                with torch.no_grad():
                    pred = net(x)
                
                # Process each sample in batch
                for j in range(x.size(0)):
                    try:
                        # Get predictions and original values
                        pred_j = pred[j].cpu().numpy()
                        orig_j = y[j].cpu().numpy()
                        
                        # Ensure predictions are in valid range
                        pred_j = np.clip(pred_j, 0, 1023)
                        orig_j = np.clip(orig_j, 0, 1023)
                        
                        # Decode audio
                        with torch.no_grad():
                            pred_audio = codec.decode(torch.from_numpy(pred_j).to(dev))
                            orig_audio = codec.decode(torch.from_numpy(orig_j).to(dev))
                        
                        # Normalize audio to prevent clipping
                        pred_audio = pred_audio / torch.max(torch.abs(pred_audio))
                        orig_audio = orig_audio / torch.max(torch.abs(orig_audio))
                        
                        # Check for audio quality
                        if torch.isnan(pred_audio).any() or torch.isinf(pred_audio).any():
                            print(f"Warning: Invalid values in predicted audio for sample {i}_{j}")
                            continue
                        
                        # Save audio files
                        pred_path = output_dir / f"pred_epoch{epoch}_sample{i}_{j}.wav"
                        orig_path = output_dir / f"orig_epoch{epoch}_sample{i}_{j}.wav"
                        
                        torchaudio.save(pred_path, pred_audio.cpu(), 48000)
                        torchaudio.save(orig_path, orig_audio.cpu(), 48000)
                        
                        # Clear memory
                        del pred_audio, orig_audio
                        torch.cuda.empty_cache()
                        
                    except Exception as e:
                        print(f"Warning: Failed to save validation audio for sample {i}_{j}: {str(e)}")
                        continue
                
                # Clear memory after each batch
                del pred, x, y, y_scales
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"Warning: Error processing batch {i}: {str(e)}")
                continue
                
    except Exception as e:
        print(f"Warning: Error in save_validation_audio: {str(e)}")
        return

def verify_ground_truth(ds, stage_dir, dev):
    """Verify ground truth audio quality."""
    print("\nVerifying ground truth audio...")
    
    # Get a few samples
    samples = []
    for i, (x, y) in enumerate(ds):
        if i >= 3:  # Check only first 3 samples
            break
        samples.append((x, y))
    
    # Decode and save
    for i, (x, y) in enumerate(samples):
        # Decode audio
        orig_audio = ds.codec.decode(y)
        
        # Save audio
        save_audio(orig_audio, stage_dir / f"gt_{i}.wav")
        
        # Clear memory
        del x, y, orig_audio
        torch.cuda.empty_cache()

def compute_audio_metrics(orig_audio, pred_audio, sr):
    """Compute audio quality metrics with proper device handling."""
    # Move to CPU and convert to numpy
    orig_audio = orig_audio.cpu().numpy()
    pred_audio = pred_audio.cpu().numpy()
    
    # Ensure audio is mono by averaging channels if needed
    if orig_audio.ndim > 1:
        orig_audio = orig_audio.mean(axis=0)
    if pred_audio.ndim > 1:
        pred_audio = pred_audio.mean(axis=0)
    
    # Compute metrics
    try:
        pesq = compute_metrics(orig_audio, pred_audio, sr)['pesq']
        stoi = compute_metrics(orig_audio, pred_audio, sr)['stoi']
    except Exception as e:
        print(f"Warning: Failed to compute metrics: {str(e)}")
        pesq, stoi = 0.0, 0.0
    
    return pesq, stoi

def compute_spectral_loss(orig_audio: torch.Tensor, 
                         pred_audio: torch.Tensor, 
                         sr: int = 48000) -> torch.Tensor:
    """Compute spectral loss between original and predicted audio.
    
    Args:
        orig_audio: Original audio tensor [B, C, T] or [B, 1, T]
        pred_audio: Predicted audio tensor [B, C, T] or [B, 1, T]
        sr: Sample rate
        
    Returns:
        Spectral loss value
    """
    # Ensure audio is mono by averaging channels if needed
    if orig_audio.shape[1] > 1:
        orig_audio = orig_audio.mean(dim=1, keepdim=True)  # [B, 1, T]
    if pred_audio.shape[1] > 1:
        pred_audio = pred_audio.mean(dim=1, keepdim=True)  # [B, 1, T]
    
    # Remove batch dimension for STFT
    orig_audio = orig_audio.squeeze(0)  # [1, T]
    pred_audio = pred_audio.squeeze(0)  # [1, T]
    
    # Compute STFT
    orig_stft = torch.stft(orig_audio, 
                          n_fft=2048, 
                          hop_length=512,
                          return_complex=True)
    pred_stft = torch.stft(pred_audio, 
                          n_fft=2048, 
                          hop_length=512,
                          return_complex=True)
    
    # Compute magnitude spectra
    orig_mag = torch.abs(orig_stft)
    pred_mag = torch.abs(pred_stft)
    
    # Compute loss on magnitude spectra
    return F.mse_loss(pred_mag, orig_mag)

def compute_loss(model: nn.Module,
                x: torch.Tensor,
                y: torch.Tensor,
                y_scales: torch.Tensor,
                codec: tp.Optional[EncodecModel] = None,
                compute_audio_loss: bool = False,
                verify_gradients: bool = False,
                stage_name: str = "stage0_identity",
                tok_batch: tp.Optional[dict] = None,
                epoch: int = 0,
                mask_target: tp.Optional[torch.Tensor] = None,
                perceptual_target: tp.Optional[torch.Tensor] = None,
                gain_target: tp.Optional[torch.Tensor] = None,
                stereo_target: tp.Optional[torch.Tensor] = None,
                compression_target: tp.Optional[torch.Tensor] = None) -> tp.Tuple[torch.Tensor, dict]:
    """
    Compute loss with optional audio loss and new auxiliary heads.
    Args:
        model: TokenUNet
        x: [B, n_q, T] degraded tokens
        y: [B, n_q, T] clean tokens
        y_scales: [B, T] scale factors
        codec: optional EncodecModel for waveform decoding
        compute_audio_loss: whether to compute waveform/audio losses
        verify_gradients: unused
        stage_name: curriculum stage
        tok_batch: optional dict of extra features
        epoch: int
        mask_target: [B, n_q, T] or None
        perceptual_target: [B, 8] or None
        gain_target: [B, 1] or None
        stereo_target: [B, 2] or None
        compression_target: [B, 2] or None
    Returns:
        total_loss, metrics_dict
    """
    outputs = model(x)
    logits = outputs['logits']  # [B, K, n_q, T]
    mask = outputs['mask']     # [B, n_q, T]
    perceptual = outputs['perceptual']  # [B, 8]
    gain = outputs['gain']     # [B, 1]
    stereo = outputs.get('stereo', None)  # [B, 2]
    compression = outputs.get('compression', None)  # [B, 2]

    # For stage 0, we only compute token-space CE loss
    if stage_name == "stage0_identity":
        logits_ = logits.permute(0, 2, 3, 1).reshape(-1, logits.size(1))  # [B*n_q*T, K]
        y_ = y.reshape(-1)  # [B*n_q*T]
        ce_loss = F.cross_entropy(logits_, y_)
        total_loss = ce_loss
        metrics_dict = {'ce_loss': ce_loss.item()}
        # Optionally add mask, perceptual, gain, stereo, compression losses if targets provided
        if mask_target is not None:
            mask_loss = F.binary_cross_entropy(mask, mask_target)
            total_loss = total_loss + 0.1 * mask_loss
            metrics_dict['mask_loss'] = mask_loss.item()
        if perceptual_target is not None:
            perceptual_loss = F.mse_loss(perceptual, perceptual_target)
            total_loss = total_loss + 0.1 * perceptual_loss
            metrics_dict['perceptual_loss'] = perceptual_loss.item()
        if gain_target is not None:
            gain_loss = F.mse_loss(gain, gain_target)
            total_loss = total_loss + 0.1 * gain_loss
            metrics_dict['gain_loss'] = gain_loss.item()
        if stereo_target is not None and stereo is not None:
            stereo_loss = F.mse_loss(stereo, stereo_target)
            total_loss = total_loss + 0.1 * stereo_loss
            metrics_dict['stereo_loss'] = stereo_loss.item()
        if compression_target is not None and compression is not None:
            compression_loss = F.mse_loss(compression, compression_target)
            total_loss = total_loss + 0.1 * compression_loss
            metrics_dict['compression_loss'] = compression_loss.item()
        return total_loss, metrics_dict
    
    # For other stages, compute full loss
    logits_ = logits.permute(0, 2, 3, 1).reshape(-1, logits.size(1))  # [B*n_q*T, K]
    y_ = y.reshape(-1)  # [B*n_q*T]
    ce_loss = F.cross_entropy(logits_, y_)
    total_loss = ce_loss
    metrics_dict = {'ce_loss': ce_loss.item()}
    # Optionally add mask, perceptual, gain, stereo, compression losses if targets provided
    if mask_target is not None:
        mask_loss = F.binary_cross_entropy(mask, mask_target)
        total_loss = total_loss + 0.1 * mask_loss
        metrics_dict['mask_loss'] = mask_loss.item()
    if perceptual_target is not None:
        perceptual_loss = F.mse_loss(perceptual, perceptual_target)
        total_loss = total_loss + 0.1 * perceptual_loss
        metrics_dict['perceptual_loss'] = perceptual_loss.item()
    if gain_target is not None:
        gain_loss = F.mse_loss(gain, gain_target)
        total_loss = total_loss + 0.1 * gain_loss
        metrics_dict['gain_loss'] = gain_loss.item()
    if stereo_target is not None and stereo is not None:
        stereo_loss = F.mse_loss(stereo, stereo_target)
        total_loss = total_loss + 0.1 * stereo_loss
        metrics_dict['stereo_loss'] = stereo_loss.item()
    if compression_target is not None and compression is not None:
        compression_loss = F.mse_loss(compression, compression_target)
        total_loss = total_loss + 0.1 * compression_loss
        metrics_dict['compression_loss'] = compression_loss.item()

    # Initialize metrics dictionary for audio losses
    metrics_dict.update({
        'audio_loss': 0.0,
        'spectral_loss': 0.0,
        'time_loss': 0.0,
        'mel_loss': 0.0,
        'token_spec_loss': 0.0
    })
    
    # Only compute audio loss if requested and not stage 0
    if compute_audio_loss and codec is not None:
        try:
            # Get predictions
            pred = logits_.argmax(dim=1)  # [B*n_q*T]
            pred = pred.reshape(x.size(0), -1, x.size(2))  # [B, n_q, T]
            
            # Decode audio
            pred_codes = list(zip(pred.unbind(0), y_scales.unbind(0)))
            orig_codes = list(zip(y.reshape(x.size(0), -1, x.size(2)).unbind(0), y_scales.unbind(0)))
            
            decoded_preds = codec.decode(pred_codes)
            decoded_orig = codec.decode(orig_codes)
            
            # Compute audio losses
            audio_loss = 0.0
            spectral_loss = 0.0
            time_loss = 0.0
            mel_loss = 0.0
            token_spec_loss = 0.0
            
            for b in range(x.size(0)):
                # Compute spectral loss
                spec_loss = compute_spectral_loss(decoded_orig[b], decoded_preds[b])
                spectral_loss += spec_loss
                
                # Compute time-domain loss
                time_loss += F.l1_loss(decoded_preds[b], decoded_orig[b])
                
                # Compute mel-spectrogram loss if available
                if tok_batch is not None and tok_batch['mel_o'] is not None:
                    mel_loss += F.mse_loss(tok_batch['mel_m'][b], tok_batch['mel_o'][b])
                
                # Compute token-space spectral loss if available
                if tok_batch is not None and tok_batch['stft_o_mag'] is not None:
                    token_spec_loss += F.mse_loss(tok_batch['stft_m_mag'][b], tok_batch['stft_o_mag'][b])
            
            # Average losses
            spectral_loss /= x.size(0)
            time_loss /= x.size(0)
            mel_loss /= x.size(0)
            token_spec_loss /= x.size(0)
            
            # Combine losses
            audio_loss = (SPECTRAL_WEIGHT * spectral_loss + 
                         TIME_WEIGHT * time_loss + 
                         AUDIO_WEIGHT * mel_loss +
                         TOKEN_SPEC_WEIGHT * token_spec_loss)
            
            # Add to total loss
            total_loss = total_loss + audio_loss
            
            # Update metrics
            metrics_dict.update({
                'audio_loss': audio_loss.item(),
                'spectral_loss': spectral_loss.item(),
                'time_loss': time_loss.item(),
                'mel_loss': mel_loss.item(),
                'token_spec_loss': token_spec_loss.item()
            })
            
        except Exception as e:
            print(f"Warning: Failed to compute audio loss: {str(e)}")
    
    return total_loss, metrics_dict

def create_run_dir(base_dir: Path) -> Path:
    """Create a unique directory for this training run."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / f"run_{timestamp}"
    run_dir.mkdir(exist_ok=True)
    return run_dir

def test_dataset_and_gradients():
    """Test dataset loading and gradient flow with a small batch."""
    print("\nRunning dataset and gradient tests...")
    
    # Set up device
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create small dataset
    ds = TokenPairDataset(
        "experiments/curriculums",
        stages=["stage0_identity"],
        force_audio=True,  # Test both paths
        model_type="48khz",
        bandwidth=24.0,
        max_debug=2,
        return_specs=True  # Test spectral features
    )
    ds.codec = ds.codec.to(dev)  # Move codec to device before any operations
    
    # Create small model
    net = TokenUNet(ds.n_q).to(dev)
    
    # Test both token and audio loading
    print("\nTesting dataset loading...")
    for i in range(min(2, len(ds))):
        X, Y, Y_scales, Y_stft, mel_spec, stft_o, stft_m, mel_o, mel_m = ds[i]
        print(f"\nSample {i}:")
        print(f"X shape: {tuple(X.shape)}")
        print(f"Y shape: {tuple(Y.shape)}")
        print(f"Y_scales shape: {tuple(Y_scales.shape)}")
        print(f"X range: [{X.min().item():.1f}, {X.max().item():.1f}]")
        print(f"Y range: [{Y.min().item():.1f}, {Y.max().item():.1f}]")
        print(f"Y_scales range: [{Y_scales.min().item():.1f}, {Y_scales.max().item():.1f}]")
        if Y_stft is not None:
            print(f"Y_stft shape: {tuple(Y_stft.shape)}")
        if mel_spec is not None:
            print(f"mel_spec shape: {tuple(mel_spec.shape)}")
        if stft_o is not None:
            print(f"stft_o shape: {tuple(stft_o.shape)}")
        if stft_m is not None:
            print(f"stft_m shape: {tuple(stft_m.shape)}")
        if mel_o is not None:
            print(f"mel_o shape: {tuple(mel_o.shape)}")
        if mel_m is not None:
            print(f"mel_m shape: {tuple(mel_m.shape)}")
        
        # Clear memory after each sample
        del X, Y, Y_scales, Y_stft, mel_spec, stft_o, stft_m, mel_o, mel_m
        torch.cuda.empty_cache()
    
    # Test gradient flow with a single sample
    print("\nTesting gradient flow...")
    # Get a single sample
    X, Y, Y_scales, Y_stft, mel_spec, stft_o, stft_m, mel_o, mel_m = ds[0]
    
    # Add batch dimension
    x = X.unsqueeze(0).to(dev)  # [1, n_q, T]
    y = Y.unsqueeze(0).to(dev)  # [1, n_q, T]
    scales = Y_scales.unsqueeze(0).to(dev)  # [1, T]
    
    # Test only token-space CE loss (no audio decode) to save memory
    try:
        loss, metrics = compute_loss(
            net, x, y, scales,
            codec=None,
            compute_audio_loss=False,
            verify_gradients=True
        )
        
        print("\nTest complete!")
        return True
    except Exception as e:
        print(f"\nError during gradient test: {str(e)}")
        return False
    finally:
        # Clean up memory
        del x, y, scales
        torch.cuda.empty_cache()

def plot_detailed_losses(metrics: dict, stage_name: str, stage_dir: Path):
    """Plot detailed loss curves with proper error handling."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f"Detailed Loss Curves - {stage_name}")
        
        # Helper function to safely plot metrics
        def safe_plot(ax, data, label, title):
            if data and len(data) > 0:
                ax.plot(data, label=label)
                ax.set_title(title)
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
                ax.legend()
                ax.grid(True)
        
        # Plot audio losses
        safe_plot(axes[0, 0], metrics.get('audio_loss', []), 'Audio Loss', 'Audio Domain Losses')
        safe_plot(axes[0, 0], metrics.get('time_loss', []), 'Time Loss', 'Audio Domain Losses')
        
        # Plot spectral losses
        safe_plot(axes[0, 1], metrics.get('spectral_loss', []), 'Spectral Loss', 'Spectral Domain Losses')
        safe_plot(axes[0, 1], metrics.get('mel_loss', []), 'Mel Loss', 'Spectral Domain Losses')
        
        # Plot token-space losses
        safe_plot(axes[0, 2], metrics.get('token_spec_loss', []), 'Token Spec Loss', 'Token-Space Losses')
        
        # Plot quality metrics
        safe_plot(axes[1, 0], metrics.get('pesq', []), 'PESQ', 'Audio Quality Metrics')
        safe_plot(axes[1, 0], metrics.get('stoi', []), 'STOI', 'Audio Quality Metrics')
        
        # Plot training dynamics
        safe_plot(axes[1, 1], metrics.get('grad_norm', []), 'Grad Norm', 'Training Dynamics')
        safe_plot(axes[1, 1], metrics.get('lr', []), 'Learning Rate', 'Training Dynamics')
        
        # Plot batch size and gradient accumulation
        safe_plot(axes[1, 2], metrics.get('batch_size', []), 'Batch Size', 'Training Configuration')
        safe_plot(axes[1, 2], metrics.get('grad_accum', []), 'Grad Accum', 'Training Configuration')
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(stage_dir / "imgs" / "detailed_losses.png")
        plt.close()
        
    except Exception as e:
        print(f"Warning: Failed to plot detailed losses: {str(e)}")
        # Try to save a simple plot as fallback
        try:
            plt.figure(figsize=(10, 6))
            plt.plot(metrics.get('train_loss', []), label='Train Loss')
            plt.plot(metrics.get('val_loss', []), label='Val Loss')
            plt.title(f"Basic Loss Curves - {stage_name}")
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            plt.savefig(stage_dir / "imgs" / "basic_losses.png")
            plt.close()
        except Exception as e2:
            print(f"Warning: Failed to save basic plot: {str(e2)}")

class OOMException(Exception):
    pass

def grad_accum_step(net, opt, scaler, x, y, y_scales, codec, compute_audio_loss, verify_gradients, stage_name, tok_batch, epoch, step, grad_accum, clip_value, dev, use_amp):
    """Single gradient accumulation step with OOM handling."""
    try:
        # Forward pass with AMP
        with autocast(device_type='cuda', enabled=use_amp):
            loss, _ = compute_loss(net, x, y, y_scales, 
                                codec=codec,
                                compute_audio_loss=compute_audio_loss,
                                verify_gradients=verify_gradients,
                                stage_name=stage_name,
                                tok_batch=tok_batch,
                                epoch=epoch,
                                stereo_target=tok_batch['stereo_target'] if tok_batch and 'stereo_target' in tok_batch else None,
                                compression_target=tok_batch['compression_target'] if tok_batch and 'compression_target' in tok_batch else None)
            loss = loss / grad_accum

        # Backward pass with AMP
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Gradient clipping
        if use_amp:
            scaler.unscale_(opt)
        grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), clip_value)
        
        # Optimizer step with AMP
        if use_amp:
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()
        opt.zero_grad()
        
        return True, False, False, None, None
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return False, True, False, max(1, x.size(0) // 2), min(16, grad_accum * 2)
        raise e
    except Exception as e:
        print(f"Warning: Error in training step: {str(e)}")
        return False, False, True, None, None

def create_loaders(train_ds, val_ds, test_ds, batch_size, dev):
    """Create data loaders with persistent workers and pin memory."""
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
        pin_memory=(dev.type == "cuda"),
        collate_fn=pad_collate
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        persistent_workers=True,
        pin_memory=(dev.type == "cuda"),
        collate_fn=pad_collate
    )
    
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        persistent_workers=True,
        pin_memory=(dev.type == "cuda"),
        collate_fn=pad_collate
    )
    
    return train_loader, val_loader, test_loader

def main():
    """Main training function."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train Token-UNet with curriculum learning")
    parser.add_argument("--until_stage", type=int, default=len(STAGES)-1,
                      help="Stop at this stage (0-based index)")
    parser.add_argument("--use_amp", action="store_true",
                      help="Enable automatic mixed precision")
    parser.add_argument("--use_checkpointing", action="store_true",
                      help="Enable gradient checkpointing")
    args = parser.parse_args()
    
    # Set up device and CUDA memory allocation
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    print(f"Using device: {dev}")
    
    # Set random seed
    set_seed(SEED)
    
    # Create output directory with timestamp
    outdir = Path("CurriculumTraining")
    outdir.mkdir(exist_ok=True)
    run_dir = create_run_dir(outdir)
    print(f"\nSaving outputs to: {run_dir}")
    
    # Training loop over stages
    for stage_idx, (stage_name, dropout, use_bottleneck, max_lr, pct_start, n_train) in enumerate(STAGES):
        if stage_idx > args.until_stage:
            break
            
        print(f"\nStarting stage {stage_idx}: {stage_name}")
        
        # Create stage directory structure under run directory
        stage_dir = run_dir / stage_name
        stage_dir.mkdir(exist_ok=True)
        (stage_dir / "ckpts").mkdir(exist_ok=True)
        (stage_dir / "imgs").mkdir(exist_ok=True)
        (stage_dir / "logs").mkdir(exist_ok=True)
        (stage_dir / "audio").mkdir(exist_ok=True)
        
        # Print stage transition info
        print_stage_transition(stage_name, dropout, use_bottleneck, max_lr, pct_start, n_train)
        
        # Stage-specific settings
        use_amp = args.use_amp and stage_name != "stage0_identity"  # Disable AMP for stage 0
        use_checkpointing = args.use_checkpointing and stage_name != "stage0_identity"  # Disable checkpointing for stage 0
        
        # Initialize metrics logging
        metrics = {
            'train_loss': [],
            'val_loss': [],
            'grad_norm': [],
            'lr': [],
            'batch_size': [],
            'clip_value': [],
            'time_per_epoch': [],
            'stage': [],
            'grad_accum': [],
            'oom': [],
            'nan': []
        }
        
        # Create CSV logger
        csv_path = stage_dir / "logs" / "training_log.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'epoch', 'train_loss', 'val_loss', 'grad_norm', 
                'lr', 'batch_size', 'clip_value', 'time', 'stage',
                'grad_accum', 'oom', 'nan'
            ])
        
        # Create dataset
        ds = TokenPairDataset(
            "experiments/curriculums",
            stages=[stage_name],
            force_audio=False,  # Always use pre-computed tokens
            model_type="48khz",
            bandwidth=24.0,
            return_specs=(stage_name != "stage0_identity")  # Only return specs for non-stage0
        )
        
        # Split dataset
        n_val = max(1, int(len(ds) * VAL_FRAC))
        n_test = max(1, int(len(ds) * TEST_FRAC))
        n_train = len(ds) - n_val - n_test
        
        train_ds, val_ds, test_ds = random_split(ds, [n_train, n_val, n_test])
        
        # Verify splits
        verify_train_val_split(ds, stage_name, stage_dir)
        
        # Stage-specific batch size and grad_accum
        if stage_name == "stage0_identity":
            batch_size = 4  # Use smaller batch size for stage 0
            grad_accum = 4  # Accumulate gradients to simulate larger batch
        else:
            batch_size = max(1, BATCH_START // 2)
            grad_accum = 2
        
        print(f"[Stage {stage_name}] batch_size={batch_size}, grad_accum={grad_accum}")
        
        # Create data loaders
        train_loader, val_loader, test_loader = create_loaders(
            train_ds, val_ds, test_ds, batch_size, dev
        )
        
        # Create model
        net = TokenUNet(
            n_q=ds.n_q,
            dropout=dropout,
            use_bottleneck=use_bottleneck,
            checkpointing=use_checkpointing
        ).to(dev)
        
        # Create optimizer
        opt = torch.optim.AdamW(net.parameters(), lr=max_lr)
        total_steps = len(train_loader) * get_epochs_per_stage(n_train)
        sched = create_scheduler(opt, total_steps, stage_name, max_lr)
        scaler = GradScaler(enabled=use_amp)
        
        # Training loop
        best_val_loss = float('inf')
        patience = 0
        
        for epoch in range(get_epochs_per_stage(n_train)):
            epoch_start = time.time()
            
            # Training
            net.train()
            total_train_loss = 0
            total_train_steps = 0
            optimizer_steps = 0
            epoch_grad_norms = []
            nan_grad_count = 0
            loss_history = []
            oom_count = 0
            MAX_OOM_ATTEMPTS = 5

            while True:  # Retry loop for OOM recovery
                try:
                    # Clear ALL per-epoch state
                    total_train_loss = 0.0
                    total_train_steps = 0
                    optimizer_steps = 0
                    epoch_grad_norms = []
                    
                    # Normal iteration over loader
                    for step, batch in enumerate(train_loader):
                        if stage_name == "stage0_identity":
                            x, y, y_scales = batch
                        else:
                            x, y, y_scales, Y_stft, mel_spec, stft_o, stft_m, mel_o, mel_m = batch
                            
                        x, y, y_scales = x.to(dev), y.to(dev), y_scales.to(dev)
                        
                        try:
                            with autocast(device_type='cuda', enabled=use_amp):
                                loss, _ = compute_loss(net, x, y, y_scales, 
                                                    codec=None,  # No codec for stage 0
                                                    compute_audio_loss=False,
                                                    verify_gradients=False,
                                                    stage_name=stage_name,
                                                    tok_batch=None,
                                                    epoch=epoch,
                                                    stereo_target=None,
                                                    compression_target=None)
                            loss = loss / grad_accum
                            
                            if loss.item() > 20.0:
                                print(f"\nWarning: Loss explosion detected at step {step}: {loss.item():.4f}")
                                print("Skipping this batch and reducing learning rate...")
                                opt.zero_grad()
                                for param_group in opt.param_groups:
                                    param_group['lr'] *= 0.5
                                print(f"New learning rate: {opt.param_groups[0]['lr']:.2e}")
                                continue
                            
                            optimizer_step, oom_triggered, nan_triggered, new_batch_size, new_grad_accum = grad_accum_step(
                                net, opt, scaler, x, y, y_scales, None, False, False, stage_name, None, epoch, step, grad_accum, CLIP_BASE, dev, use_amp
                            )
                            
                            if optimizer_step:
                                optimizer_steps += 1
                            total_train_loss += loss.item() * grad_accum
                            total_train_steps += 1
                            
                        except Exception as e:
                            if isinstance(e, OOMException):
                                raise  # Re-raise OOM to be caught by outer try/except
                            print(f"Warning: Error in training step: {str(e)}")
                            continue
                    
                    # If we get here, the epoch completed successfully
                    break
                    
                except OOMException:
                    oom_count += 1
                    if oom_count >= MAX_OOM_ATTEMPTS:
                        print(f"\nFatal: Exceeded maximum OOM attempts ({MAX_OOM_ATTEMPTS})")
                        print("Saving current state and exiting...")
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': net.state_dict(),
                            'optimizer_state_dict': opt.state_dict(),
                            'scheduler_state_dict': sched.state_dict(),
                            'train_loss': total_train_loss / total_train_steps if total_train_steps > 0 else 0.0,
                            'val_loss': avg_val_loss if 'avg_val_loss' in locals() else float('inf'),
                        }, stage_dir / 'oom_recovery.pt')
                        raise RuntimeError("Training stopped due to persistent OOM errors")
                    
                    # Clear gradients and reset scaler before retrying
                    opt.zero_grad()
                    scaler = GradScaler(enabled=use_amp)  # Reset scaler
                    torch.cuda.empty_cache()
                    
                    # Adjust batch size and grad accum
                    batch_size = max(1, batch_size // 2)
                    grad_accum = min(16, grad_accum * 2)
                    
                    # Rebuild DataLoader with new batch size
                    print(f"[OOM] Attempt {oom_count}/{MAX_OOM_ATTEMPTS}: Rebuilding DataLoader with batch_size={batch_size}, grad_accum={grad_accum}")
                    train_loader, val_loader, test_loader = create_loaders(
                        train_ds, val_ds, test_ds, batch_size, dev
                    )
                    continue  # Retry the epoch with new batch size
            
            # Compute epoch statistics
            avg_train_loss = total_train_loss / optimizer_steps if optimizer_steps > 0 else 0.0
            avg_grad_norm = sum(epoch_grad_norms) / len(epoch_grad_norms) if epoch_grad_norms else 0.0
            
            # Validation
            net.eval()
            total_val_loss = 0
            total_val_steps = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    if stage_name == "stage0_identity":
                        x, y, y_scales = batch
                    else:
                        x, y, y_scales, Y_stft, mel_spec, stft_o, stft_m, mel_o, mel_m = batch
                    
                    x, y, y_scales = x.to(dev), y.to(dev), y_scales.to(dev)
                    
                    with autocast(device_type='cuda', enabled=use_amp):
                        loss, _ = compute_loss(net, x, y, y_scales, 
                                           codec=None,  # No codec for stage 0
                                           compute_audio_loss=False,
                                           verify_gradients=False,
                                           stage_name=stage_name,
                                           tok_batch=None,
                                           epoch=epoch,
                                           stereo_target=None,
                                           compression_target=None)
                    total_val_loss += loss.item()
                    total_val_steps += 1
            
            avg_val_loss = total_val_loss / total_val_steps
            
            # Save validation audio
            if epoch % 5 == 0:  # Save audio every 5 epochs
                save_validation_audio(net, val_loader, ds.codec, epoch, stage_dir / "audio", stage_name, dev)
            
            # Update metrics
            metrics['train_loss'].append(avg_train_loss)
            metrics['val_loss'].append(avg_val_loss)
            metrics['grad_norm'].append(avg_grad_norm)
            metrics['lr'].append(opt.param_groups[0]['lr'])
            metrics['batch_size'].append(batch_size)
            metrics['clip_value'].append(CLIP_BASE)
            metrics['time_per_epoch'].append(time.time() - epoch_start)
            metrics['stage'].append(stage_idx)
            metrics['grad_accum'].append(grad_accum)
            metrics['oom'].append(oom_triggered)
            metrics['nan'].append(nan_triggered)
            
            # Log metrics
            with open(csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch, avg_train_loss, avg_val_loss, avg_grad_norm,
                    opt.param_groups[0]['lr'], batch_size, CLIP_BASE,
                    time.time() - epoch_start, stage_idx, grad_accum,
                    oom_triggered, nan_triggered
                ])
            
            # Print epoch stats
            print(f"\nEpoch {epoch}:")
            print(f"Train Loss: {avg_train_loss:.4f}")
            print(f"Val Loss: {avg_val_loss:.4f}")
            print(f"LR: {opt.param_groups[0]['lr']:.2e}")
            print(f"Time: {time.time() - epoch_start:.1f}s")
            
            # Save checkpoint
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                    'scheduler_state_dict': sched.state_dict(),
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss,
                }, stage_dir / "ckpts" / "best.pt")
            
            # Save latest checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'scheduler_state_dict': sched.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, stage_dir / "ckpts" / "latest.pt")
            
            # Plot curves
            plot_curves(metrics, stage_dir / "imgs")
            
            # Check for early stopping
            if stage_name == "stage0_identity":
                if avg_val_loss < STAGE0_VAL_THRESHOLD:
                    print(f"\nStage 0 validation loss below threshold ({STAGE0_VAL_THRESHOLD})")
                    break
            else:
                if avg_val_loss < best_val_loss:
                    patience = 0
                else:
                    patience += 1
                    if patience >= 5:
                        print("\nEarly stopping triggered")
                        break
            
            # Clear memory
            torch.cuda.empty_cache()

if __name__ == "__main__":
    # Run tests first
    if test_dataset_and_gradients():
        print("\nAll tests passed! Starting training...")
        main()
    else:
        print("\nTests failed! Please fix issues before training.")
