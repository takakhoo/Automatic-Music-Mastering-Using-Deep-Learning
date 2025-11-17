import torch
import torchaudio
import numpy as np
from pathlib import Path

def compute_metrics(orig_audio, pred_audio, sr=48000):
    """Compute audio quality metrics (PESQ and STOI) for evaluation.
    
    Args:
        orig_audio: Original audio tensor [B, C, T]
        pred_audio: Predicted audio tensor [B, C, T]
        sr: Sample rate in Hz
        
    Returns:
        Dictionary containing PESQ and STOI scores
    """
    try:
        import pesq
        import pystoi
    except ImportError:
        print("Warning: pesq or pystoi not installed. Skipping audio metrics.")
        return {'pesq': 0.0, 'stoi': 0.0}
    
    # Convert to numpy and ensure mono
    orig_np = orig_audio.squeeze().cpu().numpy()
    pred_np = pred_audio.squeeze().cpu().numpy()
    
    if len(orig_np.shape) > 1:
        orig_np = orig_np.mean(axis=0)
    if len(pred_np.shape) > 1:
        pred_np = pred_np.mean(axis=0)
    
    # Downsample to 16kHz for PESQ
    if sr == 48000:
        import scipy.signal as signal
        orig_np = signal.resample(orig_np, len(orig_np) // 3)
        pred_np = signal.resample(pred_np, len(pred_np) // 3)
        sr = 16000
    
    # Compute metrics
    try:
        pesq_score = pesq.pesq(sr, orig_np, pred_np, 'wb')
    except:
        pesq_score = 0.0
    
    try:
        stoi_score = pystoi.stoi(orig_np, pred_np, sr, extended=False)
    except:
        stoi_score = 0.0
    
    return {
        'pesq': pesq_score,
        'stoi': stoi_score
    }

def save_audio(audio, path, sr=48000):
    """Save audio tensor to file.
    
    Args:
        audio: Audio tensor [B, C, T] or [C, T]
        path: Output path
        sr: Sample rate in Hz
    """
    # Process audio (match token_baseline.py)
    if audio.dim() == 3:
        audio = audio[0].mean(dim=0)  # Collapse batch and average channels
    elif audio.dim() == 2:
        audio = audio.mean(dim=0)  # Average channels
    
    # Normalize to -1..1
    audio = audio / (audio.abs().max() + 1e-9)
    
    # Add channel dimension for torchaudio
    audio = audio.unsqueeze(0)
    
    # Create directory if needed
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save audio
    torchaudio.save(path, audio.cpu(), sr) 