import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import OneCycleLR, LinearLR, ConstantLR, SequentialLR, CosineAnnealingLR
import gc
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List
from token_dataset import TokenPairDataset

# Constants
WARM_STEPS_STAGE0 = 1000
DIV_FACTOR_OTHER = 10
MAX_PATIENCE = 6
TRAIN_PLATEAU_EPOCHS = 8
GRAD_NORM_WARN = 20

def create_scheduler(optimizer: torch.optim.Optimizer,
                    total_steps: int,
                    stage_name: str,
                    lr: float) -> torch.optim.lr_scheduler._LRScheduler:
    """Create learning rate scheduler based on stage.
    
    Args:
        optimizer: The optimizer to schedule
        total_steps: Total number of training steps
        stage_name: Current stage name
        lr: Maximum learning rate
        
    Returns:
        Configured learning rate scheduler
    """
    if stage_name == "stage0_identity":
        # OneCycleLR with longer warmup for stage 0
        return OneCycleLR(optimizer, max_lr=lr, total_steps=total_steps,
                         pct_start=0.3, div_factor=10, final_div_factor=100)
    else:
        # Cosine annealing for other stages
        return CosineAnnealingLR(optimizer, T_max=total_steps,
                                eta_min=lr/100)

def handle_oom(opt, scaler, net, dev, batch_size, grad_accum):
    """Handle out-of-memory error by adjusting batch size and gradient accumulation.
    
    Args:
        opt: Optimizer
        scaler: GradScaler (not used for update, just for reference)
        net: Model
        dev: Device
        batch_size: Current batch size
        grad_accum: Current gradient accumulation steps
        
    Returns:
        Tuple of (new_batch_size, new_grad_accum)
    """
    print("[OOM] Reducing batch size and increasing grad_accum!")
    torch.cuda.empty_cache()
    
    # Define bounds
    MIN_BATCH_SIZE = 1
    MAX_GRAD_ACCUM = 16  # Maximum gradient accumulation steps
    
    # Check if we've hit minimum batch size
    if batch_size <= MIN_BATCH_SIZE:
        if grad_accum >= MAX_GRAD_ACCUM:
            raise RuntimeError("Hit minimum batch size and maximum grad_accum - cannot reduce further")
        # Only increase grad_accum if we can't reduce batch size further
        new_batch_size = MIN_BATCH_SIZE
        new_grad_accum = min(MAX_GRAD_ACCUM, grad_accum * 2)
    else:
        # Halve batch size, double grad_accum
        new_batch_size = max(MIN_BATCH_SIZE, batch_size // 2)
        new_grad_accum = min(MAX_GRAD_ACCUM, grad_accum * 2)
    
    print(f"[OOM] New batch_size={new_batch_size}, grad_accum={new_grad_accum}")
    
    # Clear any gradients
    opt.zero_grad()
    
    return new_batch_size, new_grad_accum

def handle_nan(optimizer, lr):
    """Handle NaN loss by backing off learning rate."""
    print("NaN loss detected, reducing learning rate...")
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * 0.1
    return lr * 0.1

def compute_grad_norm_and_max(model):
    """Compute gradient norm and maximum gradient per layer."""
    total_norm = 0.0
    max_grad = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            max_grad = max(max_grad, p.grad.data.abs().max().item())
    total_norm = total_norm ** 0.5
    return total_norm, max_grad

def analyze_token_coverage(ds: TokenPairDataset,
                          stage_name: str,
                          stage_dir: Path,
                          n_samples: int = 1000) -> None:
    """Analyze token coverage in the dataset.
    
    Args:
        ds: TokenPairDataset to analyze
        stage_name: Current stage name
        stage_dir: Directory to save analysis plots
        n_samples: Number of samples to analyze
    """
    print(f"\nAnalyzing token coverage for {stage_name}...")
    
    # Collect tokens from first n_samples
    mod_tokens = []
    orig_tokens = []
    for i in range(min(n_samples, len(ds))):
        x, y, _ = ds[i]
        mod_tokens.append(x)
        orig_tokens.append(y)
    
    # Stack into tensors
    mod_tokens = torch.stack(mod_tokens)  # [N, n_q, T]
    orig_tokens = torch.stack(orig_tokens)  # [N, n_q, T]
    
    # Compute coverage statistics
    mod_coverage = torch.zeros(ds.n_q, ds.k)
    orig_coverage = torch.zeros(ds.n_q, ds.k)
    
    for q in range(ds.n_q):
        mod_coverage[q] = torch.bincount(mod_tokens[:, q].flatten(), minlength=ds.k).float()
        orig_coverage[q] = torch.bincount(orig_tokens[:, q].flatten(), minlength=ds.k).float()
    
    # Normalize
    mod_coverage /= mod_coverage.sum(dim=1, keepdim=True)
    orig_coverage /= orig_coverage.sum(dim=1, keepdim=True)
    
    # Plot coverage
    plt.figure(figsize=(15, 5))
    plt.subplot(121)
    plt.imshow(mod_coverage.cpu(), aspect='auto')
    plt.title(f"Modified Token Coverage ({stage_name})")
    plt.xlabel("Token Value")
    plt.ylabel("Codebook")
    plt.colorbar()
    
    plt.subplot(122)
    plt.imshow(orig_coverage.cpu(), aspect='auto')
    plt.title(f"Original Token Coverage ({stage_name})")
    plt.xlabel("Token Value")
    plt.ylabel("Codebook")
    plt.colorbar()
    
    plt.tight_layout()
    plt.savefig(stage_dir / "token_coverage.png")
    plt.close()
    
    # Print statistics
    print(f"Modified token coverage: {mod_coverage.mean():.3f}")
    print(f"Original token coverage: {orig_coverage.mean():.3f}")

def verify_codec_settings(ds: TokenPairDataset,
                         stage_name: str,
                         stage_dir: Path) -> None:
    """Verify EnCodec settings match expected values.
    
    Args:
        ds: TokenPairDataset to verify
        stage_name: Current stage name
        stage_dir: Directory to save verification results
    """
    print(f"\nVerifying codec settings for {stage_name}...")
    
    # Check sample rate
    assert ds.codec.sample_rate == 48000, f"Expected 48kHz, got {ds.codec.sample_rate}Hz"
    
    # Check bandwidth
    assert ds.codec.target_bandwidth == 24.0, f"Expected 24kbps, got {ds.codec.target_bandwidth}kbps"
    
    # Check number of codebooks
    assert ds.codec.quantizer.n_q == 4, f"Expected 4 codebooks, got {ds.codec.quantizer.n_q}"
    
    # Check vocabulary size
    assert ds.codec.quantizer.bins == 1024, f"Expected 1024 tokens, got {ds.codec.quantizer.bins}"
    
    # Save settings to file
    settings = {
        "sample_rate": ds.codec.sample_rate,
        "bandwidth": ds.codec.target_bandwidth,
        "n_codebooks": ds.codec.quantizer.n_q,
        "vocab_size": ds.codec.quantizer.bins,
        "stage": stage_name,
        "verified_at": datetime.now().isoformat()
    }
    
    with open(stage_dir / "codec_settings.json", "w") as f:
        json.dump(settings, f, indent=2)
    
    print("Codec settings verified and saved.")

def plot_curves(metrics: Dict[str, List[float]],
                stage_name: str,
                stage_dir: Path) -> None:
    """Plot training curves for the current stage.
    
    Args:
        metrics: Dictionary of metric names to lists of values
        stage_name: Current stage name
        stage_dir: Directory to save plots
    """
    print(f"\nPlotting training curves for {stage_name}...")
    
    # Create plots directory if it doesn't exist
    plots_dir = stage_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Plot each metric
    for metric_name, values in metrics.items():
        if not values:  # Skip empty metrics
            continue
            
        plt.figure(figsize=(10, 6))
        plt.plot(values, label=metric_name)
        plt.title(f"{metric_name} - {stage_name}")
        plt.xlabel("Step")
        plt.ylabel(metric_name)
        plt.grid(True)
        plt.legend()
        
        # Save plot
        plt.savefig(plots_dir / f"{metric_name.lower()}.png")
        plt.close()
    
    # Plot all metrics together
    plt.figure(figsize=(12, 8))
    for metric_name, values in metrics.items():
        if values:  # Only plot non-empty metrics
            plt.plot(values, label=metric_name)
    
    plt.title(f"Training Curves - {stage_name}")
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.grid(True)
    plt.legend()
    
    # Save combined plot
    plt.savefig(plots_dir / "all_metrics.png")
    plt.close()
    
    print(f"Plots saved to {plots_dir}")

class OOMException(Exception):
    """Custom exception for out-of-memory errors."""
    pass 