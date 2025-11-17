#!/usr/bin/env python3
"""
Test script for TokenUNet - verifies token prediction and optionally audio reconstruction
"""

from __future__ import annotations
import torch
from token_dataset import TokenPairDataset
from token_unet import TokenUNet
from encodec import EncodecModel
from encodec.utils import convert_audio
import torchaudio
import soundfile as sf
from pathlib import Path
import numpy as np

def analyze_audio(wav: torch.Tensor, name: str):
    """Analyze audio properties to verify quality."""
    wav = wav.squeeze().numpy()
    print(f"\n{name} Analysis:")
    print(f"Shape: {wav.shape}")
    print(f"Range: [{wav.min():.3f}, {wav.max():.3f}]")
    print(f"Mean: {wav.mean():.3f}")
    print(f"Std: {wav.std():.3f}")
    print(f"Zero crossings: {np.sum(np.diff(np.signbit(wav)))}")
    print(f"Silence ratio: {(np.abs(wav) < 0.01).mean():.2%}")

def test_unet(stage: str = "stage1_single", sample_idx: int = 0, save_audio: bool = True):
    """Test the UNet's ability to transform degraded audio back to original.
    
    Args:
        stage: Which curriculum stage to test (e.g., "stage1_single")
        sample_idx: Which sample to test
        save_audio: Whether to save audio files for comparison
    """
    # Initialize dataset and model
    ds = TokenPairDataset("experiments/curriculums", stages=[stage])
    model = TokenUNet(ds.n_q)
    model.eval()  # Set to evaluation mode
    
    # Get a sample
    X, Y = ds[sample_idx]  # X=degraded, Y=original
    X = X.unsqueeze(0)  # Add batch dimension
    
    # Forward pass
    with torch.no_grad():
        logits = model(X)  # [1, K, n_q, T]
        pred = logits.argmax(dim=1)  # [1, n_q, T]
    
    # Verify token validity
    K = logits.shape[1]  # Number of possible tokens per codebook
    valid_tokens = (pred >= 0) & (pred < K)
    valid_ratio = valid_tokens.float().mean()
    print(f"\nToken validity check:")
    print(f"Valid token ratio: {valid_ratio:.2%}")
    print(f"Token range: [{pred.min().item()}, {pred.max().item()}]")
    print(f"Expected range: [0, {K-1}]")
    
    # Compare predictions with original (Y)
    correct = (pred == Y.unsqueeze(0)).float().mean()
    print(f"\nToken prediction accuracy: {correct:.2%}")
    
    if save_audio:
        # Initialize EnCodec
        codec = EncodecModel.encodec_model_48khz()
        codec.set_target_bandwidth(24.0)
        
        # Get the original WAV file path
        if hasattr(ds, 'clean_wav') and ds.clean_wav:
            # If we have direct WAV access
            clean = ds.clean_wav[sample_idx]
            print(f"\nLoading original WAV: {clean}")
            orig_wav, sr = torchaudio.load(clean)
            print(f"Original WAV shape: {orig_wav.shape}")
        else:
            # If we're using precomputed tokens
            print("\nUsing precomputed tokens - no direct WAV access")
            orig_wav = None
            sr = 48000
        
        # Decode tokens back to audio
        with torch.no_grad():
            # Degraded input
            X_reshaped = X.squeeze(0).unsqueeze(0)  # Add batch dim back
            degraded_audio = codec.decode([(X_reshaped, None)])[0]
            
            # Original target
            Y_reshaped = Y.unsqueeze(0)  # Add batch dim
            original_audio = codec.decode([(Y_reshaped, None)])[0]
            
            # Model prediction
            pred_reshaped = pred.squeeze(0).unsqueeze(0)  # Add batch dim back
            predicted_audio = codec.decode([(pred_reshaped, None)])[0]
        
        # Process audio (match token_baseline.py)
        def process_audio(wav):
            # Collapse to mono if needed
            if wav.ndim == 3:
                wav = wav[0].mean(dim=0)
            elif wav.ndim == 2:
                wav = wav.mean(dim=0)
            # Normalize to -1..1
            wav = wav / (wav.abs().max() + 1e-9)
            return wav.unsqueeze(0)  # Add channel dimension for torchaudio
        
        # Analyze audio quality
        analyze_audio(degraded_audio, "Degraded Input")
        analyze_audio(original_audio, "Original Target")
        analyze_audio(predicted_audio, "Model Prediction")
        
        # Save audio files
        out_dir = Path("test_outputs")
        out_dir.mkdir(exist_ok=True)
        
        # Process and save all versions
        if orig_wav is not None:
            torchaudio.save(out_dir / "original_direct.wav", orig_wav, sr)
        torchaudio.save(out_dir / "degraded.wav", process_audio(degraded_audio), 48000)
        torchaudio.save(out_dir / "original_tokens.wav", process_audio(original_audio), 48000)
        torchaudio.save(out_dir / "predicted.wav", process_audio(predicted_audio), 48000)
        print(f"\nSaved audio files to {out_dir}/")
        if orig_wav is not None:
            print("original_direct.wav - Original WAV from dataset")
        print("degraded.wav - Input degraded audio")
        print("original_tokens.wav - Target original audio")
        print("predicted.wav - Model's attempt to restore the audio")

if __name__ == "__main__":
    # Test with stage1_single (single effect) by default
    test_unet(stage="stage1_single", sample_idx=0, save_audio=True) 