# src/token_dataset.py  –  paired EnCodec token dataset
# --------------------------------------------------------------------------------
# This file implements a PyTorch Dataset for handling paired audio data
# where each sample consists of a degraded audio and its original version.
# The dataset converts audio waveforms into EnCodec tokens for training.

from __future__ import annotations
import pathlib, typing as tp, torch, torchaudio
from torch.utils.data import Dataset
from encodec import EncodecModel
from encodec.utils import convert_audio
import random
from token_constants import PAD  # Import PAD from constants
import torch.nn.functional as F

# Type aliases for better type hints
TokenTuple = tp.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]  # (X, Y, scales)
SpecTuple = tp.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, tp.Optional[torch.Tensor], tp.Optional[torch.Tensor], tp.Optional[torch.Tensor], tp.Optional[torch.Tensor], tp.Optional[torch.Tensor], tp.Optional[torch.Tensor]]  # (X, Y, scales, Y_stft, mel_spec, stft_o, stft_m, mel_o, mel_m)
MetaTuple = tp.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]  # (X, Y, scales, meta)
SpecMetaTuple = tp.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, tp.Optional[torch.Tensor], tp.Optional[torch.Tensor], tp.Optional[torch.Tensor], tp.Optional[torch.Tensor], tp.Optional[torch.Tensor], tp.Optional[torch.Tensor], dict]  # (X, Y, scales, Y_stft, mel_spec, stft_o, stft_m, mel_o, mel_m, meta)
DatasetOutput = tp.Union[TokenTuple, SpecTuple, MetaTuple, SpecMetaTuple]

# Type aliases for collated batches
CollatedTokenTuple = tp.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]  # (B×X, B×Y, B×scales)
CollatedSpecTuple = tp.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, tp.Optional[torch.Tensor], tp.Optional[torch.Tensor], tp.Optional[torch.Tensor], tp.Optional[torch.Tensor], tp.Optional[torch.Tensor], tp.Optional[torch.Tensor]]  # (B×X, B×Y, B×scales, B×Y_stft, B×mel_spec, B×stft_o, B×stft_m, B×mel_o, B×mel_m)
CollatedMetaTuple = tp.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, tp.List[dict]]  # (B×X, B×Y, B×scales, [meta])
CollatedSpecMetaTuple = tp.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, tp.Optional[torch.Tensor], tp.Optional[torch.Tensor], tp.Optional[torch.Tensor], tp.Optional[torch.Tensor], tp.Optional[torch.Tensor], tp.Optional[torch.Tensor], tp.List[dict]]  # (B×X, B×Y, B×scales, B×Y_stft, B×mel_spec, B×stft_o, B×stft_m, B×mel_o, B×mel_m, [meta])
CollatedOutput = tp.Union[CollatedTokenTuple, CollatedSpecTuple, CollatedMetaTuple, CollatedSpecMetaTuple]

# ──────────────────────────────────────────────────────────────────────
def _load_codec(model_type: str, bandwidth: float) -> EncodecModel:
    """Initialize and configure the EnCodec model for audio encoding/decoding.
    
    Args:
        model_type: Either "24khz" or "48khz" to specify the model variant
        bandwidth: Target bandwidth in kbps (must be valid for the model type)
    
    Returns:
        Configured EnCodec model instance
    
    Raises:
        ValueError: If model_type or bandwidth is invalid
    """
    if model_type == "24khz":
        codec = EncodecModel.encodec_model_24khz()
        allowed = {1.5, 3, 6, 12, 24}
    elif model_type == "48khz":
        codec = EncodecModel.encodec_model_48khz()
        allowed = {3, 6, 12, 24}
    else:
        raise ValueError("model_type must be '24khz' or '48khz'")
    if bandwidth not in allowed:
        raise ValueError(f"{bandwidth} kbps invalid for {model_type}")
    codec.set_target_bandwidth(bandwidth)
    return codec

# ──────────────────────────────────────────────────────────────────────
class TokenPairDataset(Dataset):
    """Dataset for paired audio data (degraded + original) using EnCodec tokens.
    
    This dataset loads pairs of audio files from curriculum stages where:
    - The degraded version has "_modified.wav" suffix
    - The original version has "_original.wav" suffix
    
    The audio is converted to EnCodec tokens for training the Token-UNet model.
    Scales are loaded on the fly from the original audio files to preserve quality.
    
    When loading from precomputed .pt files, additional metadata is available:
    - STFT and mel spectrograms (if return_specs=True)
    - STFT configuration and metadata
    - Debug ranges for tokens and scales
    - Cached waveforms (if available)
    """
    
    def __init__(self,
                base_dir    : str | pathlib.Path,
                stages      : list[str] | None = None,
                model_type  : str   = "48khz",
                bandwidth   : float = 24.0,
                force_audio : bool  = False,
                return_specs: bool  = False,
                return_meta : bool  = False,
                max_debug   : int   = 3):
        """Initialize the dataset.
        
        Args:
            base_dir: Base directory containing curriculum stages
            stages: List of stage names to include (e.g., ["stage0_identity", "stage1_single"])
                   If None, uses all available stages
            model_type: EnCodec model type ("24khz" or "48khz")
            bandwidth: Target bandwidth in kbps
            force_audio: If True, always load from audio files instead of tokens
            return_specs: If True, return STFT and mel spectrograms when available
            return_meta: If True, return metadata and configuration info
            max_debug: Number of samples to print debug info for
        """
        self.base_dir = pathlib.Path(base_dir)
        if not self.base_dir.exists():
            raise RuntimeError(f"Base directory {self.base_dir} not found")

        # Find all available stages if none specified
        if stages is None:
            self.stages = [d.name for d in self.base_dir.iterdir() 
                         if d.is_dir() and d.name.startswith("stage")]
        else:
            self.stages = stages

        # Check if precomputed tokens exist for the first stage
        self.token_files = []
        self.clean_wav = []
        for stage in self.stages:
            stage_dir = self.base_dir / stage
            token_dir = stage_dir / "output_tokens"
            audio_dir = stage_dir / "output_audio"
            if not force_audio and token_dir.exists() and any(token_dir.glob("*.pt")):
                stage_token_files = sorted(token_dir.glob("*.pt"))
                self.token_files.extend(stage_token_files)
                print(f"[TokenDataset] Using precomputed tokens for {stage} ({len(stage_token_files)} files)")
            elif audio_dir.exists():
                stage_files = sorted(audio_dir.glob("*_original.wav"))
                self.clean_wav.extend(stage_files)
                print(f"[TokenDataset] Using audio for {stage} ({len(stage_files)} files)")
            else:
                print(f"Warning: Stage directory {stage_dir} missing output_audio and output_tokens")

        if not self.token_files and not self.clean_wav:
            raise RuntimeError(f"No token files or audio files found in any of {self.stages}")

        # Initialize EnCodec for scale extraction
        self.codec = _load_codec(model_type, bandwidth)
        if torch.cuda.is_available():
            self.codec = self.codec.to(torch.device("cuda"))
        self.sr = self.codec.sample_rate
        self.channels = self.codec.channels
        self.n_q = self.codec.quantizer.n_q  # Number of codebooks
        self._dbg_left = int(max_debug)
        self.return_specs = return_specs
        self.return_meta = return_meta

        print(f"[TokenDataset] model {model_type}  • {bandwidth} kbps")
        print(f"[TokenDataset] {self.sr} Hz  {self.channels}-ch  "
            f"{self.n_q} codebooks")
        print(f"[TokenDataset] Using stages: {', '.join(self.stages)}")

        if self.token_files:
            random.shuffle(self.token_files)  # Shuffle for random_split
        else:
            random.shuffle(self.clean_wav)    # Shuffle for random_split

    def _wav2codes(self, path: pathlib.Path) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert audio file to EnCodec tokens and scales.
        
        Args:
            path: Path to the audio file
            
        Returns:
            Tuple of (tokens, scales) where:
            - tokens is of shape [n_q, T]
            - scales is of shape [T]
        """
        # Load and preprocess audio
        wav, sr = torchaudio.load(path)
        wav = convert_audio(wav, sr, self.sr, self.channels).unsqueeze(0)
        wav = wav.to(next(self.codec.parameters()).device)  # Move to same device as codec
        
        # Encode to tokens and scales
        with torch.no_grad():
            frames = self.codec.encode(wav)  # list[(codes,scale)]
        
        # Get the total length from the first frame
        total_len = frames[0][0].shape[-1] * len(frames)
        
        # Pre-allocate tensors
        codes = torch.zeros(self.n_q, total_len, dtype=torch.long, device=wav.device)
        scales = torch.ones(total_len, device=wav.device)  # Initialize with ones instead of zeros
        
        # Fill in the tensors frame by frame
        current_pos = 0
        for frame_codes, frame_scales in frames:
            frame_len = frame_codes.shape[-1]
            codes[:, current_pos:current_pos + frame_len] = frame_codes.squeeze(0)
            
            # Handle scales properly
            if frame_scales.dim() == 0:  # Single value for the frame
                frame_scales = frame_scales.expand(frame_len)  # Expand to frame length
            elif frame_scales.dim() == 1:  # Already a tensor of shape [frame_len]
                frame_scales = frame_scales.squeeze(0)  # Remove batch dim if present
            
            # Ensure scales are non-zero and reasonable
            frame_scales = torch.clamp(frame_scales, min=1e-6, max=1.0)  # Clamp to reasonable range
            
            scales[current_pos:current_pos + frame_len] = frame_scales
            current_pos += frame_len
        
        # Final check for zero scales
        if scales.min() < 1e-6:
            print(f"Warning: Found zero scales in {path}")
            scales = torch.clamp(scales, min=1e-6)
        
        return codes, scales  # [n_q,T], [T]

    def __getitem__(self, idx: int) -> DatasetOutput:
        """Get a pair of degraded and original audio tokens with scales.
        
        Args:
            idx: Index of the sample to retrieve
            
        Returns:
            If return_specs and return_meta are False:
                Tuple of (degraded_tokens, original_tokens, original_scales)
            If return_specs is True and return_meta is False:
                Tuple of (degraded_tokens, original_tokens, original_scales, Y_stft, mel_spec, stft_o, stft_m, mel_o, mel_m)
            If return_meta is True:
                Additional metadata dict as the last element
        """
        if self.token_files:
            tok_file = self.token_files[idx]
            # Load to CPU to prevent device mismatches
            try:
                tok = torch.load(tok_file, map_location="cpu", weights_only=True)
            except TypeError:  # PyTorch < 2.0
                tok = torch.load(tok_file, map_location="cpu")
            X, Y = tok["X"], tok["Y"]
            # Token integrity check
            assert X.shape[0] == self.n_q and Y.shape[0] == self.n_q, f"Token shape mismatch in {tok_file}"
            
            # Use precomputed scales if available, else fallback
            if "scales" in tok:
                Y_scales = tok["scales"]
            else:
                stem = tok_file.stem
                clean = tok_file.parent.parent / "output_audio" / f"{stem}_original.wav"
                _, Y_scales = self._wav2codes(clean)
            
            if self._dbg_left > 0:
                self._dbg_left -= 1
                print(f"[{idx}] {tok_file.name} (tokens) → X {tuple(X.shape)} Y {tuple(Y.shape)}")
            
            # Build return tuple based on requested features
            result = [X, Y, Y_scales]
            
            # Add spectrograms and audio-based spectra if requested
            if getattr(self, "return_specs", False):
                # Token-based
                Y_stft = tok.get("Y_stft_mag", None)
                mel_spec = tok.get("mel_spec", None)
                # Audio-based
                stft_o = tok.get("stft_o_mag", None)
                stft_m = tok.get("stft_m_mag", None)
                mel_o = tok.get("mel_o", None)
                mel_m = tok.get("mel_m", None)
                result.extend([Y_stft, mel_spec, stft_o, stft_m, mel_o, mel_m])
            
            # Add metadata if requested
            if getattr(self, "return_meta", False):
                meta = {
                    "ranges": tok.get("ranges", None),
                    "stft_cfg": tok.get("stft_cfg", None),
                    "metadata": tok.get("metadata", None),
                    "wav_o": tok.get("wav_o", None),
                    "wav_m": tok.get("wav_m", None)
                }
                result.append(meta)
            
            return tuple(result)
            
        # Get file paths
        clean = self.clean_wav[idx]
        stem = clean.stem.split("_")[0]
        deg = clean.parent / f"{stem}_modified.wav"
        if not deg.exists():
            raise FileNotFoundError(f"Missing degraded {deg}")

        # Convert both files to tokens and scales
        X, _ = self._wav2codes(deg)
        Y, Y_scales = self._wav2codes(clean)

        # Debug output for first few samples
        if self._dbg_left > 0:
            self._dbg_left -= 1
            stage = clean.parent.parent.name
            print(f"[{idx}] {stage}/{deg.name} → {tuple(X.shape)} "
                f"min {X.min().item()}  max {X.max().item()}")
            print(f"     {stage}/{clean.name} → {tuple(Y.shape)} "
                f"min {Y.min().item()}  max {Y.max().item()}\n")
        
        # Build return tuple based on requested features
        result = [X, Y, Y_scales]
        
        # Add None for spectrograms when loading from audio
        if getattr(self, "return_specs", False):
            result.extend([None, None, None, None, None, None])  # 6 None values for all spectral features
        
        # Add empty metadata when loading from audio
        if getattr(self, "return_meta", False):
            meta = {
                "ranges": None,
                "stft_cfg": None,
                "metadata": None,
                "wav_o": None,
                "wav_m": None
            }
            result.append(meta)
        
        return tuple(result)

    def __len__(self):
        # Return correct length for both token and audio modes
        return len(self.token_files) if self.token_files else len(self.clean_wav)

# ---------------------------------------------------------------------
def pad_scales(tensor: torch.Tensor, T: int) -> torch.Tensor:
    """Pad 1-D or 2-D scale tensor by repeating its last element.
    
    Args:
        tensor: Scale tensor of shape [T] or [..., T]
        T: Target length
        
    Returns:
        Padded tensor of same shape as input but with length T
    """
    L = tensor.shape[-1]
    if L >= T:
        return tensor
    pad_amount = T - L
    last = tensor[..., -1:]
    
    # Handle 1-D case
    if tensor.ndim == 1:
        return torch.cat([tensor, last.repeat(pad_amount)], dim=0)
    
    # Handle multi-dimensional case
    return torch.cat([tensor, last.expand(*tensor.shape[:-1], pad_amount)], dim=-1)

def pad_spec(spec: torch.Tensor, T_max: int) -> torch.Tensor:
    """Pad a spectrogram tensor to match the maximum time dimension.
    
    Args:
        spec: Spectrogram tensor of shape [C, F, T] or [F, T]
        T_max: Target time dimension length
        
    Returns:
        Padded tensor with same shape as input but with time dimension T_max
    """
    if spec.shape[-1] >= T_max:
        return spec
    pad_amt = T_max - spec.shape[-1]
    # Handle both 2D and 3D tensors
    if spec.ndim == 2:
        return F.pad(spec, (0, pad_amt))  # [F, T]
    return F.pad(spec, (0, pad_amt, 0, 0, 0, 0))  # [C, F, T]

def pad_collate(batch: tp.List[DatasetOutput], pad_val: int = PAD) -> CollatedOutput:
    """Collate function for DataLoader to handle variable length sequences.
    
    This function pads all sequences in a batch to the length of the longest one.
    Tokens are padded with PAD value, while scales are padded by repeating the last valid value.
    Spectrograms are padded to match the maximum time dimension in the batch.
    
    Args:
        batch: List of tuples containing:
            - degraded_tokens
            - original_tokens
            - original_scales
            - Y_stft_mag (optional)
            - mel_spec (optional)
            - stft_o_mag (optional)
            - stft_m_mag (optional)
            - mel_o (optional)
            - mel_m (optional)
            - metadata (optional)
        pad_val: Value to use for padding tokens (default: PAD from token_unet)
        
    Returns:
        Tuple containing padded tensors and metadata in the same order as input
    """
    # Determine what features are present
    has_specs = len(batch[0]) >= 5  # True if we have specs (even if they're None)
    has_meta = len(batch[0]) >= 10  # Updated for new spectral features
    
    # Extract base tensors (X, Y, scales)
    xs, ys, zs = zip(*[b[:3] for b in batch])
    T = max(t.shape[-1] for t in xs)
    
    # Pad tokens with PAD value
    pad = lambda t: torch.nn.functional.pad(t, (0, T - t.shape[-1]),
                                          value=pad_val)
    x_padded = torch.stack([pad(x) for x in xs])
    y_padded = torch.stack([pad(y) for y in ys])
    
    # Pad scales by repeating last value
    z_padded = torch.stack([pad_scales(z, T) for z in zs])
    
    # Build result tuple
    result = [x_padded, y_padded, z_padded]
    
    # Handle spectrograms and audio-based spectra if present
    if has_specs:
        # Original token-based
        stft_tok = [b[3] for b in batch]
        mel_tok = [b[4] for b in batch]
        # Audio-based
        stft_o = [b[5] for b in batch]
        stft_m = [b[6] for b in batch]
        mel_o = [b[7] for b in batch]
        mel_m = [b[8] for b in batch]
        
        # Find maximum time dimension for each type of spectrogram
        T_stft = max(s.shape[-1] for s in stft_tok if s is not None) if any(s is not None for s in stft_tok) else 0
        T_mel = max(s.shape[-1] for s in mel_tok if s is not None) if any(s is not None for s in mel_tok) else 0
        T_stft_o = max(s.shape[-1] for s in stft_o if s is not None) if any(s is not None for s in stft_o) else 0
        T_stft_m = max(s.shape[-1] for s in stft_m if s is not None) if any(s is not None for s in stft_m) else 0
        T_mel_o = max(s.shape[-1] for s in mel_o if s is not None) if any(s is not None for s in mel_o) else 0
        T_mel_m = max(s.shape[-1] for s in mel_m if s is not None) if any(s is not None for s in mel_m) else 0
        
        # Stack non-None values with padding
        def stack_or_none(lst, T_max):
            if not any(x is not None for x in lst):
                return None
            padded = [pad_spec(x, T_max) for x in lst if x is not None]
            return torch.stack(padded) if padded else None
        
        result.extend([
            stack_or_none(stft_tok, T_stft),
            stack_or_none(mel_tok, T_mel),
            stack_or_none(stft_o, T_stft_o),
            stack_or_none(stft_m, T_stft_m),
            stack_or_none(mel_o, T_mel_o),
            stack_or_none(mel_m, T_mel_m)
        ])
    
    # Handle metadata if present
    if has_meta:
        # Metadata doesn't need padding, just pass through
        result.append([b[-1] for b in batch])
    
    return tuple(result)

# ---------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, required=True)
    parser.add_argument("--stage", type=str, required=True)
    parser.add_argument("--model_type", type=str, default="48khz")
    parser.add_argument("--bandwidth", type=float, default=24.0)
    parser.add_argument("--force_audio", action="store_true")
    args = parser.parse_args()

    # Create dataset
    ds = TokenPairDataset(
        base_dir=args.base_dir,
        stages=[args.stage],
        model_type=args.model_type,
        bandwidth=args.bandwidth,
        force_audio=args.force_audio,
        max_debug=2
    )
    print(f"\nDataset size: {len(ds)}")

    # Test first few samples
    for i in range(min(3, len(ds))):
        X, Y, Y_scales = ds[i]
        print(f"\nSample {i}:")
        print(f"X shape: {tuple(X.shape)}")
        print(f"Y shape: {tuple(Y.shape)}")
        print(f"Y_scales shape: {tuple(Y_scales.shape)}")
        print(f"X range: [{X.min().item():.1f}, {X.max().item():.1f}]")
        print(f"Y range: [{Y.min().item():.1f}, {Y.max().item():.1f}]")
        print(f"Y_scales range: [{Y_scales.min().item():.1f}, {Y_scales.max().item():.1f}]")
