#!/usr/bin/env python3
"""
Quick baseline: decode EnCodec tokens from .pt files and save audio for listening.

Usage:
    python src/token_baseline.py --stage stage0_identity --num 5

This script loads .pt token files from a curriculum stage's output_tokens directory,
decodes both X (degraded) and Y (original) tokens to audio, and saves them as .wav files
in a new 'Baseline Test' directory for quick listening checks.
"""
import os, torch, argparse
from pathlib import Path
from encodec import EncodecModel
import torchaudio
from token_dataset import TokenPairDataset, pad_collate
from token_constants import PAD  # Import PAD from constants

def verify_padding(tokens: torch.Tensor):
    """Verify that padding is consistent with PAD constant."""
    if tokens.eq(PAD).any():
        print(f"Found {tokens.eq(PAD).sum().item()} padded positions")
        # Check if padding is only at the end
        last_non_pad = (tokens != PAD).long().argmax(dim=-1)
        if (tokens[..., last_non_pad:] == PAD).all():
            print("✓ Padding is properly placed at sequence ends")
        else:
            print("⚠ Warning: Found PAD tokens in middle of sequences")
    else:
        print("No padding found in tokens")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=str, required=True, help='Curriculum stage (e.g. stage0_identity)')
    parser.add_argument('--base_dir', type=str, default='experiments/curriculums', help='Base curriculums dir')
    parser.add_argument('--num', type=int, default=5, help='Number of examples to decode')
    parser.add_argument('--out_dir', type=str, default='Baseline Test', help='Output directory for decoded audio')
    args = parser.parse_args()

    # Create dataset to get scales
    ds = TokenPairDataset(
        args.base_dir,
        stages=[args.stage],
        model_type="48khz",
        bandwidth=24.0
    )
    
    stage_dir = Path(args.base_dir) / args.stage / 'output_tokens'
    if not stage_dir.exists():
        raise FileNotFoundError(f"Token directory not found: {stage_dir}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load EnCodec model (48kHz, 24kbps)
    codec = EncodecModel.encodec_model_48khz()
    codec.set_target_bandwidth(24.0)
    codec.eval()

    pt_files = sorted(stage_dir.glob('*.pt'))[:args.num]
    if not pt_files:
        raise RuntimeError(f"No .pt files found in {stage_dir}")

    print(f"Decoding {len(pt_files)} token files from {stage_dir}...")
    for i, pt_path in enumerate(pt_files):
        data = torch.load(pt_path, map_location='cpu')
        
        # Verify padding in both X and Y
        print(f"\nVerifying padding in {pt_path.name}:")
        print("X tokens:")
        verify_padding(data['X'])
        print("Y tokens:")
        verify_padding(data['Y'])
        
        # Get scales from dataset
        stem = pt_path.stem
        clean = pt_path.parent.parent / "output_audio" / f"{stem}_original.wav"
        _, scales = ds._wav2codes(clean)
        
        for key in ['X', 'Y']:
            tokens = data[key].unsqueeze(0)  # [1, n_q, T]
            # Decode with scales
            with torch.no_grad():
                wav = codec.decode([(tokens, scales)])[0].cpu()
            # Collapse to mono if needed
            if wav.ndim == 3:
                wav = wav[0].mean(dim=0)
            elif wav.ndim == 2:
                wav = wav.mean(dim=0)
            # Normalize to -1..1
            wav = wav / (wav.abs().max() + 1e-9)
            out_path = out_dir / f"{pt_path.stem}_{key}.wav"
            torchaudio.save(str(out_path), wav.unsqueeze(0), codec.sample_rate)
            print(f"[{i+1}/{len(pt_files)}] Saved: {out_path}")
    print(f"\n✓ All decoded audio saved in: {out_dir.resolve()}")

if __name__ == "__main__":
    main() 