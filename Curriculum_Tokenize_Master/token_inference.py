#!/usr/bin/env python3
"""
token_inference.py – Inference for Token-UNet on precomputed EnCodec tokens

- Loads a trained TokenUNet checkpoint
- Processes all .pt token pairs in a given stage's output_tokens/
- Decodes both original and model-restored audio using EnCodec
- Saves audio, mel/chroma images, and CSV metrics to out_dir
"""

import argparse
import csv
import sys
from pathlib import Path
from tqdm import tqdm
import torch, torchaudio
import numpy as np
import librosa, librosa.display, matplotlib.pyplot as plt
import json

from encodec import EncodecModel
from token_unet import TokenUNet
from token_dataset import TokenPairDataset

# ──────────────── Helper Functions ────────────────

def save_wav(wav, sr, path):
    wav = np.clip(wav, -1, 1)
    torchaudio.save(str(path), torch.from_numpy(wav).unsqueeze(0), sr)

def save_mel(wav, sr, path, title):
    mel = librosa.feature.melspectrogram(y=wav, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
    dB  = librosa.power_to_db(mel, ref=np.max)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(dB, sr=sr, hop_length=512, x_axis="time", y_axis="mel", cmap="magma")
    plt.colorbar(format='%+2.0f dB'); plt.title(title); plt.tight_layout()
    plt.savefig(path, dpi=120); plt.close()

def save_chroma(wav, sr, path, title):
    chr_ = librosa.feature.chroma_stft(y=wav, sr=sr, n_fft=2048, hop_length=512)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(chr_, x_axis="time", y_axis="chroma", cmap="coolwarm")
    plt.colorbar(); plt.title(title); plt.tight_layout()
    plt.savefig(path, dpi=120); plt.close()

def metrics(orig: np.ndarray, rest: np.ndarray, sr_orig: int):
    """Return SNR, PESQ, STOI (resample to 16 kHz if needed)."""
    from pesq import pesq
    from pystoi import stoi
    if orig.ndim > 1:  orig  = orig.mean(0)
    if rest.ndim > 1:  rest  = rest.mean(0)
    orig /= np.max(np.abs(orig) + 1e-9)
    rest /= np.max(np.abs(rest) + 1e-9)
    snr = 10 * np.log10(np.sum(orig**2) / np.sum((orig-rest)**2) + 1e-9)
    if sr_orig != 16_000:
        orig_rs = librosa.resample(orig, orig_sr=sr_orig, target_sr=16_000)
        rest_rs = librosa.resample(rest, orig_sr=sr_orig, target_sr=16_000)
        sr_m    = 16_000
    else:
        orig_rs, rest_rs, sr_m = orig, rest, sr_orig
    pesq_s = pesq(sr_m, orig_rs, rest_rs, 'wb')
    stoi_s = stoi(orig_rs, rest_rs, sr_m, extended=False)
    return snr, pesq_s, stoi_s

# Helper to ensure mono audio for metrics
def to_mono(wav):
    return wav.mean(axis=0) if wav.ndim > 1 else wav

# Spectral convergence and log-MSE metrics
def spectral_convergence(S_ref, S_est):
    # S_ref, S_est: [freq, time] (magnitude)
    return np.linalg.norm(S_ref - S_est, 'fro') / (np.linalg.norm(S_ref, 'fro') + 1e-8)

def log_mse(S_ref, S_est):
    # S_ref, S_est: [freq, time] (magnitude)
    log_S_ref = np.log(np.abs(S_ref) + 1e-8)
    log_S_est = np.log(np.abs(S_est) + 1e-8)
    return np.mean((log_S_ref - log_S_est) ** 2)

def align_lengths(a, b):
    L = min(len(a), len(b))
    return a[:L], b[:L]

# ──────────────── Main Inference ────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=Path, help="Path to model checkpoint (.pt)")
    ap.add_argument("--stage", required=True, type=str, help="Which stage to test (e.g. stage1_single)")
    ap.add_argument("--base_dir", required=True, type=Path, help="Root experiments/curriculums directory")
    ap.add_argument("--out_dir", required=False, type=Path, default=Path("CurriculumInference"), help="Output directory for results")
    ap.add_argument("--codec", choices=("24khz","48khz"), default="48khz")
    ap.add_argument("--bandwd", type=float, default=24.0)
    ap.add_argument("--num_examples", type=int, default=2, help="Number of audio pairs to evaluate")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    codec = EncodecModel.encodec_model_48khz() if args.codec == "48khz" else EncodecModel.encodec_model_24khz()
    codec.set_target_bandwidth(args.bandwd)
    codec = codec.to(dev).eval()
    sr = codec.sample_rate
    print(f"✓ EnCodec {args.codec} {args.bandwd} kbps  sr={sr}")

    # Helper closes over local codec
    def decode_codes(codes_cuda):
        arr = codec.decode([(codes_cuda, None)])[0].cpu()
        # Process audio (match token_baseline.py)
        if arr.dim() == 3:
            arr = arr[0].mean(dim=0)  # Collapse batch and average channels
        elif arr.dim() == 2:
            arr = arr.mean(dim=0)  # Average channels
        # Normalize to -1..1
        arr = arr / (arr.abs().max() + 1e-9)
        return arr.numpy()  # shape (T,)

    # Load model
    ckpt = torch.load(args.ckpt, map_location=dev)
    if "model" not in ckpt:
        raise KeyError("Checkpoint is missing 'model' state_dict")

    # Load model config for this run
    config_path = args.ckpt.parent.parent.parent / "config.json"
    with open(config_path) as f:
        config = json.load(f)
    # Extract per-stage settings
    stage_settings = {s[0]: s for s in config["stages"]}
    if args.stage not in stage_settings:
        raise ValueError(f"Stage {args.stage} not found in config.json")
    dropout = stage_settings[args.stage][1]
    use_bottleneck = stage_settings[args.stage][2]
    # Get n_q from TokenPairDataset (matches training logic)
    ds = TokenPairDataset(
        args.base_dir,
        stages=[args.stage],
        model_type=args.codec,
        bandwidth=args.bandwd,
        max_debug=args.num_examples,   # force it to include the info dict
    )
    n_q = ds.n_q
    # Instantiate model with correct hyperparameters
    net = TokenUNet(n_q, dropout=dropout, use_bottleneck=use_bottleneck).to(dev)
    # Load checkpoint strictly (all keys must match)
    net.load_state_dict(ckpt["model"])
    net.eval()
    print(f"✓ Loaded checkpoint {args.ckpt}")

    # Directory setup
    stage_dir = args.base_dir / args.stage
    out_dir = args.out_dir / args.stage
    for d in (
        out_dir/"audio_input", out_dir/"audio_gt", out_dir/"audio_pred",
        out_dir/"audio_gt_rec",  # Add directory for GT token reconstruction
        out_dir/"images"/"mel_input", out_dir/"images"/"mel_gt", out_dir/"images"/"mel_pred",
        out_dir/"images"/"chroma_input", out_dir/"images"/"chroma_gt", out_dir/"images"/"chroma_pred",
    ):
        d.mkdir(parents=True, exist_ok=True)

    qual = [("file","snr","pesq","stoi","spec_conv","log_mse")]
    spec_conv_vals, log_mse_vals, snr_vals, pesq_vals, stoi_vals = [], [], [], [], []

    # Process examples (max_debug already limits to num_examples)
    for i, item in enumerate(ds):
        # Handle both (X, Y) and (X, Y, info) cases
        if len(item) == 3:
            X, Y, info = item
            # Get stem from pt_path in info dict
            stem = Path(info["pt_path"]).stem
            print(f"[{i}] {stem}.pt (tokens) → X {X.shape} Y {Y.shape}")
        else:
            X, Y = item
            # Only use example_XXX if we don't have the info dict
            stem = f"example_{i:03d}"
            print(f"[{i}] {stem}.pt (tokens) → X {X.shape} Y {Y.shape}")

        Xc = X.unsqueeze(0).to(dev)
        Yc = Y.unsqueeze(0).to(dev)

        # 1) model forward
        logits = net(Xc)              # [1,K,n_q,T]
        Yhat  = logits.argmax(1)      # [1,n_q,T]

        # 2a) decode the model prediction
        wav_pred = decode_codes(Yhat)
        # 2b) decode the ground-truth tokens so we can sanity-check the codec
        wav_gt_rec = decode_codes(Yc)

        # save the GT-reconstruction as well
        save_wav(wav_gt_rec, sr, out_dir/"audio_gt_rec"/f"{stem}_gt_rec.wav")

        # 3) load the real before/after files from output_audio
        raw_wav_dir = stage_dir / "output_audio"
        try:
            wav_mod,  _ = librosa.load(raw_wav_dir / f"{stem}_modified.wav", sr=sr, mono=True)
            wav_orig, _ = librosa.load(raw_wav_dir / f"{stem}_original.wav", sr=sr, mono=True)
        except FileNotFoundError as e:
            print(f"Warning: Could not find audio files for {stem}: {e}")
            print(f"Looking in: {raw_wav_dir}")
            continue

        # 4) clamp lengths and level match
        wav_pred, wav_orig = align_lengths(wav_pred, wav_orig)
        wav_mod, _ = align_lengths(wav_mod, wav_orig)
        wav_pred *= np.max(np.abs(wav_orig)) / (np.max(np.abs(wav_pred)) + 1e-9)

        # 5) save the three audio streams
        save_wav(wav_mod,  sr, out_dir/"audio_input"/f"{stem}_input.wav")
        save_wav(wav_orig, sr, out_dir/"audio_gt"/   f"{stem}_gt.wav")
        save_wav(wav_pred, sr, out_dir/"audio_pred"/ f"{stem}_pred.wav")

        # 4) dump spectrograms / chroma for each
        save_mel (wav_mod,  sr, out_dir/"images"/"mel_input"/ f"{stem}_mel_input.png",  f"{stem} – mel INPUT")
        save_mel (wav_orig, sr, out_dir/"images"/"mel_gt"/    f"{stem}_mel_gt.png",     f"{stem} – mel GT")
        save_mel (wav_pred, sr, out_dir/"images"/"mel_pred"/  f"{stem}_mel_pred.png",   f"{stem} – mel PRED")

        save_chroma(wav_mod,  sr, out_dir/"images"/"chroma_input"/ f"{stem}_chr_input.png",  f"{stem} – chroma INPUT")
        save_chroma(wav_orig, sr, out_dir/"images"/"chroma_gt"/    f"{stem}_chr_gt.png",     f"{stem} – chroma GT")
        save_chroma(wav_pred, sr, out_dir/"images"/"chroma_pred"/  f"{stem}_chr_pred.png",   f"{stem} – chroma PRED")

        # 5) metrics should be on (GT vs PRED), not (INPUT vs PRED)
        try:
            mel_gt   = librosa.feature.melspectrogram(y=to_mono(wav_orig), sr=sr, n_fft=2048, hop_length=512, n_mels=128)
            mel_pred = librosa.feature.melspectrogram(y=to_mono(wav_pred), sr=sr, n_fft=2048, hop_length=512, n_mels=128)
            spec_conv = spectral_convergence(mel_gt, mel_pred)
            logmse = log_mse(mel_gt, mel_pred)
            snr, pesq_s, stoi_s = metrics(to_mono(wav_orig), to_mono(wav_pred), sr)
            qual.append((stem, f"{snr:.2f}", f"{pesq_s:.2f}", f"{stoi_s:.2f}", f"{spec_conv:.4f}", f"{logmse:.4f}"))
            print(f"   {stem}: SNR {snr:.2f} dB  PESQ {pesq_s:.2f}  STOI {stoi_s:.2f}  SC {spec_conv:.4f}  logMSE {logmse:.4f}")
            snr_vals.append(snr)
            pesq_vals.append(pesq_s)
            stoi_vals.append(stoi_s)
            spec_conv_vals.append(spec_conv)
            log_mse_vals.append(logmse)
        except Exception as e:
            print(f"   {stem}: Metric error: {e}")

    # Write CSVs with utf-8 encoding
    with open(out_dir / "quality_metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f); writer.writerows(qual)
    # Generate summary bar plot
    metrics_names = ["SNR", "PESQ", "STOI", "Spectral Convergence", "log-MSE"]
    means = [
        np.mean(snr_vals) if snr_vals else 0,
        np.mean(pesq_vals) if pesq_vals else 0,
        np.mean(stoi_vals) if stoi_vals else 0,
        np.mean(spec_conv_vals) if spec_conv_vals else 0,
        np.mean(log_mse_vals) if log_mse_vals else 0,
    ]
    plt.figure(figsize=(8,5))
    plt.bar(metrics_names, means, color=["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"])
    plt.ylabel("Average Value")
    plt.title(f"Average Metrics Across All Samples ({args.stage})")
    plt.tight_layout()
    plt.savefig(out_dir / "summary_metrics.png", dpi=120)
    plt.close()
    print(f"\n✓ Results and summary plot saved in {out_dir}\n")

if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
