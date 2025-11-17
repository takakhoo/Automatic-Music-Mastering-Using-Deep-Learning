# ------------------------------------------------------------
# src/token_unet.py – 1-D U-Net for EnCodec tokens (Curriculum Ready)
# ------------------------------------------------------------
# Redesigned for curriculum learning: memory-efficient, robust, and modular.
# Includes CBAM, FiLM, learnable skip gating, and transposed conv upsampling.
#
# Refinements:
#   - Optional bottleneck (1x1 Conv1d) after mid block (use_bottleneck)
#   - Configurable dropout rate (dropout)
#   - set_dropout() method for dynamic adjustment
#
# When to use options:
#   - use_bottleneck=True: Only if you see poor generalization or loss spikes in stage 4/4-stronger (set from training script)
#   - dropout > 0.10: If you see overfitting in later stages, especially stage 4 (set from training script or via set_dropout)
#   - Gradient norm printing: Should be handled in the training script, not here
#   - Checkpointing: Leave off unless you hit OOM (set from training script)
#
# Control all options from the training script for flexibility.

from __future__ import annotations
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.checkpoint import checkpoint  # Explicit import for clarity
from einops import rearrange
from token_dataset import TokenPairDataset, pad_collate
from token_constants import PAD  # Import PAD from constants
import math  # For ceil division

# ─── Model hyper-parameters ──────────────────────────────────
BASE_DIM, DEPTH, K = 384, 4, 1024  # Base channels, depth, vocab size

# ─── CBAM: Convolutional Block Attention Module ─────────────
class CBAM(nn.Module):
    """Lightweight attention: channel + spatial attention."""
    def __init__(self, ch, reduction=16, kernel=7):
        super().__init__()
        # Channel attention
        self.mlp = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(ch, ch // reduction, 1), nn.ReLU(),
            nn.Conv1d(ch // reduction, ch, 1)
        )
        # Spatial attention
        self.conv = nn.Conv1d(2, 1, kernel, padding=kernel//2)
    def forward(self, x):
        # Channel attention
        w = torch.sigmoid(self.mlp(x))
        x = x * w
        # Spatial attention
        max_out, _ = torch.max(x, 1, keepdim=True)
        avg_out = torch.mean(x, 1, keepdim=True)
        s = torch.cat([max_out, avg_out], dim=1)
        s = torch.sigmoid(self.conv(s))
        return x * s

# ─── FiLM: Feature-wise Linear Modulation ───────────────────
class FiLM(nn.Module):
    """Learnable per-channel scale and shift."""
    def __init__(self, ch):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, ch, 1))
        self.beta = nn.Parameter(torch.zeros(1, ch, 1))
    def forward(self, x):
        return x * self.gamma + self.beta

# ─── Residual Block ─────────────────────────────────────────
class ResBlock(nn.Module):
    """Residual block: conv → norm → GELU → dropout → conv → norm → GELU."""
    def __init__(self, ch, dropout=0.10):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(ch, ch, 3, padding=1, bias=False),
            nn.GroupNorm(8, ch),  # Replace BatchNorm with GroupNorm
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(ch, ch, 3, padding=1, bias=False),
            nn.GroupNorm(8, ch),  # Replace BatchNorm with GroupNorm
            nn.GELU()
        )
    def forward(self, x):
        return x + self.body(x)

# ─── U-Net main class ───────────────────────────────────────
class TokenUNet(nn.Module):
    """
    1D U-Net for EnCodec tokens, curriculum-ready.
    - 4 encoder/decoder blocks, base_dim=384, doubling channels
    - CBAM and FiLM for attention and modulation
    - ConvTranspose1d for upsampling
    - Learnable skip gating
    - Optional bottleneck after mid block
    - Configurable dropout rate
    - Robust to different curriculum stages
    """
    def __init__(self, n_q: int, k: int = K, base_dim: int = BASE_DIM, depth: int = DEPTH,
                 checkpointing: bool = False, use_bottleneck: bool = False, dropout: float = 0.10):
        super().__init__()
        self.n_q, self.k, self.depth = n_q, k, depth
        self.checkpointing = checkpointing
        self.use_bottleneck = use_bottleneck
        self.dropout = dropout
        
        # Ensure embedding dimension is divisible by n_q
        emb_q = math.ceil(base_dim / n_q)  # Round up to ensure enough capacity
        self.emb = nn.Embedding(n_q * k, emb_q)
        self.inp = nn.Conv1d(n_q * emb_q, base_dim, 1)  # Project to exact base_dim
        
        # Encoder: 2x ResBlock + CBAM + downsample (Conv1d)
        ch, enc = base_dim, nn.ModuleList()
        self.skip_gates = nn.ParameterList()
        for _ in range(depth):
            enc.append(nn.Sequential(
                ResBlock(ch, dropout=dropout),
                CBAM(ch),
                ResBlock(ch, dropout=dropout),
                CBAM(ch),
                nn.Conv1d(ch, ch*2, 4, stride=2, padding=1)
            ))
            self.skip_gates.append(nn.Parameter(torch.tensor(1.0)))  # Learnable skip gate
            ch *= 2
        self.enc = enc
        
        # Mid block: ResBlock + Dropout + FiLM
        self.mid = nn.Sequential(
            ResBlock(ch, dropout=dropout),
            nn.Dropout(dropout),
            FiLM(ch)
        )
        
        # Minimal temporal context block (dilated Conv1d) for dereverberation/echo
        self.temporal_context = nn.Conv1d(ch, ch, kernel_size=9, padding=8, dilation=2, groups=1)
        
        # Optional bottleneck after mid block
        if use_bottleneck:
            self.bottleneck = nn.Conv1d(ch, ch, 1)
        else:
            self.bottleneck = None
            
        # Decoder: upsample (ConvTranspose1d) + 2x ResBlock + CBAM
        dec = nn.ModuleList()
        for _ in range(depth):
            dec.append(nn.Sequential(
                nn.ConvTranspose1d(ch, ch//2, 4, stride=2, padding=1),
                ResBlock(ch//2, dropout=dropout),
                CBAM(ch//2),
                ResBlock(ch//2, dropout=dropout),
                CBAM(ch//2)
            ))
            ch //= 2
        self.dec = dec
        
        # Prediction heads: one per codebook
        self.heads = nn.ModuleList([
            nn.Conv1d(base_dim, k, 1) for _ in range(n_q)
        ])
        # Mask head for soft mask prediction (for dereverb/echo/denoise)
        self.mask_head = nn.Conv1d(base_dim, n_q, 1)
        # Perceptual parameter head (for spectral matching/EQ/decompression)
        self.perceptual_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(base_dim, 8)
        )
        # Global gain head (for gain estimation)
        self.gain_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(base_dim, 1)
        )
        # Stereo width head (optional, for stereo enhancement / separation)
        self.stereo_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(base_dim, 2)  # [stereo width, interchannel phase]
        )
        # Compression dynamics head (e.g., predicts loudness variance or crest factor)
        self.compression_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(base_dim, 2)  # [RMS deviation, crest factor]
        )
        self.apply(self._init_w)

    @staticmethod
    def _init_w(m: nn.Module):
        """Kaiming initialization for conv/linear layers."""
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.kaiming_uniform_(m.weight, a=.2, mode="fan_in", nonlinearity="leaky_relu")
            if m.bias is not None: nn.init.zeros_(m.bias)

    @staticmethod
    def _crop(t: torch.Tensor, tgt: int):
        """Center crop for skip connections if needed."""
        diff = t.size(-1) - tgt
        if diff <= 0: return t
        lo = diff // 2
        return t[..., lo:lo + tgt]

    def set_dropout(self, new_dropout: float):
        """Dynamically set dropout rate for all ResBlocks and mid Dropout."""
        self.dropout = new_dropout
        # Update encoder
        for blk in self.enc:
            for layer in blk:
                if isinstance(layer, ResBlock):
                    for m in layer.body:
                        if isinstance(m, nn.Dropout):
                            m.p = new_dropout
        # Update mid block
        for m in self.mid:
            if isinstance(m, nn.Dropout):
                m.p = new_dropout
            if isinstance(m, ResBlock):
                for mm in m.body:
                    if isinstance(mm, nn.Dropout):
                        mm.p = new_dropout
        # Update decoder
        for blk in self.dec:
            for layer in blk:
                if isinstance(layer, ResBlock):
                    for m in layer.body:
                        if isinstance(m, nn.Dropout):
                            m.p = new_dropout

    def forward(self, tok: torch.LongTensor):
        """
        Args:
            tok: [B, n_q, T] EnCodec token indices
        Returns:
            dict with:
                'logits': [B, K, n_q, T] unnormalized probabilities
                'mask': [B, n_q, T] soft mask (0-1)
                'perceptual': [B, 8] perceptual parameter vector
                'gain': [B, 1] global gain estimate
                'stereo': [B, 2] stereo width and interchannel phase
                'compression': [B, 2] RMS deviation and crest factor
        """
        B, n_q, T = tok.shape
        print(f"[Input] tok: {tok.shape} (B, n_q, T)")
        assert n_q == self.n_q, f"Expected {self.n_q} codebooks, got {n_q}"
        
        # Create mask for padding tokens and zero them out before embedding
        pad_mask = tok.eq(PAD)
        tok_safe = tok.masked_fill(pad_mask, 0)  # Zero out padded positions
        print(f"[After PAD mask] tok_safe: {tok_safe.shape}")
        
        # Token embedding with offset
        offset = (torch.arange(n_q, device=tok.device) * self.k)[None, :, None]
        idx = offset + tok_safe  # Add offset after zeroing padded positions
        x = rearrange(self.emb(idx), 'b q t d -> b (q d) t')
        print(f"[After Embedding+Rearrange] x: {x.shape} (B, n_q*emb_q, T)")
        x = self.inp(x)  # Project to exact base_dim
        print(f"[After Input Projection] x: {x.shape} (B, base_dim, T)")
        
        # Encoder with gradient checkpointing
        skips = []
        for i, blk in enumerate(self.enc):
            x_in = x
            print(f"[Encoder {i+1} Input] {x_in.shape}")
            if self.checkpointing:
                x = checkpoint(blk, x)
            else:
                x = blk(x)
            print(f"[Encoder {i+1} Output] {x.shape}")
            skips.append(x)
        
        # Mid block with gradient checkpointing
        x_in = x
        if self.checkpointing:
            x = checkpoint(self.mid, x)
        else:
            x = self.mid(x)
        print(f"[Mid Block] in: {x_in.shape} out: {x.shape}")
        
        # Temporal context block (dilated Conv1d)
        x_in = x
        x = self.temporal_context(x)
        print(f"[Temporal Context] in: {x_in.shape} out: {x.shape}")
        
        # Optional bottleneck
        if self.bottleneck is not None:
            x_in = x
            x = self.bottleneck(x)
            print(f"[Bottleneck] in: {x_in.shape} out: {x.shape}")
        
        # Decoder with gradient checkpointing
        for i, blk in enumerate(self.dec):
            skip = skips.pop()
            gate = torch.sigmoid(self.skip_gates[-(i+1)])
            skip_cropped = self._crop(skip, x.size(-1))
            print(f"[Decoder {i+1} skip] {skip.shape} cropped: {skip_cropped.shape} gate: {gate.item():.4f}")
            x_in = x
            print(f"[Decoder {i+1} Input] {x_in.shape}")
            if self.checkpointing:
                x = checkpoint(blk, x + gate * skip_cropped)
            else:
                x = blk(x + gate * skip_cropped)
            print(f"[Decoder {i+1} Output] {x.shape}")
        
        # Ensure output length matches input
        if x.size(-1) != T:
            print(f"[Interpolate] x: {x.shape} -> T: {T}")
            x = F.interpolate(x, size=T, mode='nearest')
            print(f"[After Interpolate] x: {x.shape}")
        
        # Mask head (soft mask)
        mask = torch.sigmoid(self.mask_head(x))  # [B, n_q, T]
        # Perceptual parameter head
        perceptual_params = self.perceptual_head(x)  # [B, 8]
        # Global gain head
        gain = self.gain_head(x)  # [B, 1]
        # Stereo head output
        stereo = self.stereo_head(x)  # [B, 2]
        # Compression head output
        compression = self.compression_head(x)  # [B, 2]
        
        # Prediction heads (one per codebook)
        B, C, T = x.shape
        K = self.k
        logits = []
        for qi, head in enumerate(self.heads):
            head_out = head(x)              # [B, K, T]
            print(f"[Head {qi}] in: {x.shape} out: {head_out.shape}")
            logits.append(head_out.unsqueeze(2))   # keep 3-D
        out = torch.cat(logits, dim=2)            # (B, K, n_q, T)
        print(f"[Output] logits: {out.shape} (B, K, n_q, T)")
        return {'logits': out, 'mask': mask, 'perceptual': perceptual_params, 'gain': gain, 'stereo': stereo, 'compression': compression}

    def get_skip_gate_penalty(self) -> torch.Tensor:
        """Compute L2 penalty on skip gates for regularization."""
        return torch.sum(torch.stack([gate ** 2 for gate in self.skip_gates]))

def print_model_stats(model):
    """Prints parameter count for the model."""
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Model Stats] Total parameters: {n_params:,}")

# ────── Self-test: run "python src/token_unet.py" ───────────
if __name__ == "__main__":
    """
    Quick test of the model's shape transformations and block connections.
    Loads a small batch from TokenPairDataset and runs a forward pass.
    To test with bottleneck or different dropout, edit below:
        net = TokenUNet(ds.n_q, use_bottleneck=True, dropout=0.15)
    """
    print("▶ quick shape-check (curriculum-ready TokenUNet)")
    base_dir = sys.argv[1] if len(sys.argv) > 1 else "thesis_project/experiments/curriculums"
    ds = TokenPairDataset(base_dir, stages=["stage3_triple"], model_type="48khz", bandwidth=24.0, max_debug=1)
    dl = DataLoader(ds, 2, collate_fn=pad_collate, num_workers=0)
    x, y, scales = next(iter(dl))  # Unpack all three values
    # Edit here to test with/without bottleneck and different dropout
    net = TokenUNet(ds.n_q, use_bottleneck=False, dropout=0.10)
    print_model_stats(net)
    with torch.no_grad():
        logits = net(x)
        print(f"[test] input: {tuple(x.shape)}  logits: {tuple(logits['logits'].shape)}")
        print("Model test completed. If no errors, block connections are stable.")
