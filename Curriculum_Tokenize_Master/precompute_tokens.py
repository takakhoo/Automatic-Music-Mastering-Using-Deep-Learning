#!/usr/bin/env python3
"""
Precompute EnCodec tokens, scales, STFT magnitudes and mel-specs for every pair of
*_original.wav / *_modified.wav, and dump them into .pt files.
This script will update existing .pt files with new features if they exist.

Features:
- Backwards-compatible updates of existing .pt files
- FP16 storage for spectral features to save memory
- Comprehensive metadata and shape validation
- Robust error handling and logging
- Parallel processing across CPU cores
- Configurable waveform caching

Output .pt file structure:
{
    'X':           torch.LongTensor[n_q, T],      # Degraded tokens
    'Y':           torch.LongTensor[n_q, T],      # Clean tokens
    'scales':      torch.FloatTensor[T],          # Per-frame scale factors
    'Y_stft_mag':  torch.FloatTensor[n_q, F, T],  # STFT magnitudes (FP16)
    'mel_spec':    torch.FloatTensor[M, T],       # Mel spectrogram (FP16, optional)
    'ranges': {                                   # Min/max values for debugging
        'X': (min, max),
        'Y': (min, max),
        'scales': (min, max)
    },
    'stft_cfg': {                                 # STFT configuration
        'n_fft': int,
        'hop_length': int,
        'window': str,
        'n_freq_bins': int,
        'n_time_frames': int
    },
    'metadata': {                                 # File provenance
        'model_type': str,
        'bandwidth': float,
        'n_q': int,
        'sample_rate': int,
        'channels': int,
        'original_audio_hash': str,
        'modified_audio_hash': str,
        'processed_at': str,
        'version': str
    },
    'wav_o':       torch.FloatTensor[1, T],       # Original waveform (if cached)
    'wav_m':       torch.FloatTensor[1, T]        # Modified waveform (if cached)
}
"""

import torch
import torchaudio
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from encodec import EncodecModel
from encodec.utils import convert_audio
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import datetime
import warnings
import json
import hashlib
import os

# ────────────── CONFIG ────────────────────────────────────────────────
CURRICULUM_DIR = Path("experiments/curriculums")
BANDWIDTH      = 24.0
MODEL_TYPE     = "48khz"
N_FFT          = 2048
HOP_LENGTH     = 512
USE_MEL        = True    # Enable mel-specs for potential mel-based losses
CACHE_WAV      = True    # Cache waveforms for faster reprocessing
HASH_CHUNK     = 1024*1024  # 1MB chunks for file hashing
# ─────────────────────────────────────────────────────────────────────

# one‐time setups
codec = EncodecModel.encodec_model_48khz() if MODEL_TYPE=="48khz" else EncodecModel.encodec_model_24khz()
codec.set_target_bandwidth(BANDWIDTH)

mel_transform = None
if USE_MEL:
    mel_transform = torch.nn.Sequential(
        MelSpectrogram(sample_rate=codec.sample_rate, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=128),
        AmplitudeToDB()
    )

# Create Hann window once
hann_window = torch.hann_window(N_FFT)

def compute_file_hash(path: Path, chunk_size: int = HASH_CHUNK) -> str:
    """Compute SHA256 hash of file using first and last chunks for efficiency.
    
    For small files (≤2*chunk_size), computes full hash.
    For large files, only hashes first and last chunk_size bytes.
    """
    size = os.path.getsize(path)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        if size <= 2*chunk_size:
            for block in iter(lambda: f.read(4096), b""):
                h.update(block)
        else:
            h.update(f.read(chunk_size))
            f.seek(size - chunk_size)
            h.update(f.read(chunk_size))
    return h.hexdigest()

def validate_tensor_shapes(data: dict, expected_shapes: dict) -> bool:
    """Validate tensor shapes in the data dictionary, using -1 as wildcard.
    
    Args:
        data: Dictionary of tensors to validate
        expected_shapes: Dictionary mapping keys to expected shapes.
                        Use -1 for variable-length dimensions.
    
    Returns:
        bool: True if all shapes match their expected patterns
    """
    for key, exp_shape in expected_shapes.items():
        if key not in data:
            return False
        act_shape = tuple(data[key].shape)
        if len(act_shape) != len(exp_shape):
            return False
        for a, e in zip(act_shape, exp_shape):
            if e != -1 and a != e:
                return False
    return True

def process_stage(stage_dir: Path):
    """Process all audio pairs in a stage directory."""
    audio_dir = stage_dir / "output_audio"
    token_dir = stage_dir / "output_tokens"
    token_dir.mkdir(exist_ok=True, parents=True)
    log_path = stage_dir / "tokens_log.txt"
    log_lines = []

    def work(orig_path: Path):
        """Process a single audio pair."""
        sid = orig_path.stem.split("_")[0]
        mod_path = audio_dir / f"{sid}_modified.wav"
        out_path = token_dir / f"{sid}.pt"
        
        try:
            # Load existing data if available
            existing_data = {}
            if out_path.exists():
                try:
                    existing_data = torch.load(out_path)
                    print(f"[*] Updating existing file: {out_path.name}")
                except Exception as e:
                    warnings.warn(f"Failed to load existing file {out_path}: {e}")
                    existing_data = {}
            
            # 1) load & convert (or use cached waveforms)
            if CACHE_WAV and "wav_o" in existing_data and "wav_m" in existing_data:
                wav_o = existing_data["wav_o"]
                wav_m = existing_data["wav_m"]
            else:
                wav_o, sr = torchaudio.load(orig_path)
                wav_m, _  = torchaudio.load(mod_path)
                wav_o = convert_audio(wav_o, sr, codec.sample_rate, codec.channels).unsqueeze(0)
                wav_m = convert_audio(wav_m, sr, codec.sample_rate, codec.channels).unsqueeze(0)

            with torch.no_grad():
                # 2) encode tokens (only if not in existing data)
                if "X" not in existing_data or "Y" not in existing_data:
                    frames_o = codec.encode(wav_o)
                    frames_m = codec.encode(wav_m)
                    Y = torch.cat([f[0] for f in frames_o], dim=-1).squeeze(0).long()   # [n_q, T]
                    X = torch.cat([f[0] for f in frames_m], dim=-1).squeeze(0).long()  # [n_q, T]
                else:
                    X = existing_data["X"]
                    Y = existing_data["Y"]
                    frames_o = codec.encode(wav_o)  # Still need frames for scales

                # 3) extract scales
                scales = torch.cat([f[1].expand(1, f[0].shape[-1]) for f in frames_o], dim=-1).squeeze(0)

            # 4) precompute STFT magnitude of the *clean* token‐sequence
            Y_float = Y.float()  # [n_q, T]
            stft_mag = []
            for q in range(Y_float.shape[0]):
                spec = torch.stft(
                    Y_float[q],
                    n_fft=N_FFT,
                    hop_length=HOP_LENGTH,
                    window=hann_window.to(Y_float.device),
                    return_complex=True
                )
                stft_mag.append(torch.abs(spec))
            Y_stft_mag = torch.stack(stft_mag, 0)  # [n_q, freq_bins, time_frames]
            Y_stft_mag = Y_stft_mag.half()  # Convert to FP16 to save memory

            # 5) optional: mel-spectrogram on the *true* clean audio
            mel_spec = None
            if USE_MEL:
                mel_spec = mel_transform(wav_o.squeeze(0))  # [n_mels, time_frames]
                mel_spec = mel_spec.half()  # Convert to FP16 to save memory

            # 6) record comprehensive metadata
            meta = {
                "X":           X.cpu(),
                "Y":           Y.cpu(),
                "scales":      scales.cpu(),
                "ranges": {
                    "X":     (float(X.min()), float(X.max())),
                    "Y":     (float(Y.min()), float(Y.max())),
                    "scales":(float(scales.min()), float(scales.max())),
                },
                "Y_stft_mag":  Y_stft_mag.cpu(),
                "mel_spec":    mel_spec.cpu() if mel_spec is not None else None,
                "stft_cfg":    {
                    "n_fft": N_FFT,
                    "hop_length": HOP_LENGTH,
                    "window": "hann",
                    "n_freq_bins": Y_stft_mag.shape[1],
                    "n_time_frames": Y_stft_mag.shape[2]
                },
                "metadata": {
                    "model_type": MODEL_TYPE,
                    "bandwidth": BANDWIDTH,
                    "n_q": codec.quantizer.n_q,
                    "sample_rate": codec.sample_rate,
                    "channels": codec.channels,
                    "original_audio_hash": compute_file_hash(orig_path),
                    "modified_audio_hash": compute_file_hash(mod_path),
                    "processed_at": datetime.datetime.now().isoformat(),
                    "version": "1.0"  # For future compatibility
                }
            }

            # Cache waveforms if enabled
            if CACHE_WAV:
                meta["wav_o"] = wav_o.cpu()
                meta["wav_m"] = wav_m.cpu()

            # Update existing data with new features
            existing_data.update(meta)
            
            # Validate shapes before saving
            expected_shapes = {
                "X": (codec.quantizer.n_q, -1),
                "Y": (codec.quantizer.n_q, -1),
                "scales": (-1,),
                "Y_stft_mag": (codec.quantizer.n_q, N_FFT//2 + 1, -1)
            }
            if not validate_tensor_shapes(existing_data, expected_shapes):
                raise ValueError(f"Shape validation failed for {out_path}")
            
            # Save with error handling
            try:
                torch.save(existing_data, out_path)
            except Exception as e:
                # If save fails, try to save to a backup file
                backup_path = out_path.with_suffix('.pt.bak')
                torch.save(existing_data, backup_path)
                raise RuntimeError(f"Failed to save {out_path}, backup saved to {backup_path}: {e}")
            
            msg = f"{datetime.datetime.now()}  OK: {out_path.relative_to(stage_dir.parent.parent)}"
            print(f"[✓] {out_path.relative_to(stage_dir.parent.parent)}")
            return msg

        except Exception as e:
            warnings.warn(f"Failed to process {sid}: {e}")
            return f"{datetime.datetime.now()}  WARN: Failed to process {sid}: {e}"

    # Process all files in parallel
    wavs = sorted(audio_dir.glob("*_original.wav"))
    with ThreadPoolExecutor() as exe:
        for result in tqdm(exe.map(work, wavs), total=len(wavs), desc=f"Tokenizing {stage_dir.name}"):
            if result:
                log_lines.append(result)

    # Write log file
    with open(log_path, "a") as f:
        f.write(f"\n{datetime.datetime.now()}  Processed {len(wavs)} files\n")
        for line in log_lines:
            f.write(line + "\n")

if __name__=="__main__":
    print(f"\nConfiguration:")
    print(f"- Model: {MODEL_TYPE} at {BANDWIDTH} kbps")
    print(f"- STFT: n_fft={N_FFT}, hop_length={HOP_LENGTH}")
    print(f"- Mel-spec: {'enabled' if USE_MEL else 'disabled'}")
    print(f"- Waveform caching: {'enabled' if CACHE_WAV else 'disabled'}")
    print(f"- File hash chunk size: {HASH_CHUNK//1024//1024}MB")
    
    for stage in sorted(CURRICULUM_DIR.iterdir()):
        if (stage/"output_audio").is_dir():
            print(f"\n🔁 Processing {stage.name}...")
            process_stage(stage)
    print("\n✅ Token precomputation complete.") 