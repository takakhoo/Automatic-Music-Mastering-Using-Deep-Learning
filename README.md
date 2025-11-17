# Token U-Net: Neural Audio Codec Remastering for Full-Mix Music Restoration

**Honors Thesis Project for ENGS 88 at Dartmouth College**  
**Author:** Taka Khoo  
**Primary Advisor:** Peter Chin  
**Secondary Consultant:** Michael Casey

---

## 🎯 Architecture Overview

The Token U-Net is a 1.08 billion parameter neural network architecture designed for token-to-token audio enhancement. Below is the complete architecture diagram:

![Token U-Net Architecture](unet.png)

*Complete Token U-Net architecture with encoder-decoder structure, CBAM attention, FiLM modulation, gated skip connections, and multi-head outputs for token prediction and auxiliary tasks.*

---

## 📁 Source Code

**⚠️ IMPORTANT: The main, most recent, and complete source code is located in the `Curriculum_Tokenize_Master/` folder.**

This folder contains the full implementation of:
- ✅ EnCodec tokenization pipeline
- ✅ Curriculum-aware training system
- ✅ Complete Token U-Net architecture with attention mechanisms
- ✅ All auxiliary heads (mask, gain, compression, stereo, perceptual)
- ✅ Precomputation utilities for efficient training
- ✅ Inference and evaluation scripts

**All other folders (`src/`, `CBAMFiLMUNet + InvLSTM src/`, `DeepUnet & LSTM src/`, etc.) are experimental baselines and earlier iterations.**

---

## Table of Contents

1. [Introduction](#introduction)
2. [Problem Statement](#problem-statement)
3. [Architecture Overview](#-architecture-overview)
4. [Mathematical Foundations](#mathematical-foundations)
5. [Dataset and Preprocessing](#dataset-and-preprocessing)
6. [Training Pipeline](#training-pipeline)
7. [Evaluation and Results](#evaluation-and-results)
8. [Code Structure](#code-structure)
9. [Usage Guide](#usage-guide)
10. [Future Work](#future-work)

---

## Introduction

### Motivation

Over the past decade, music production has become increasingly democratized. Affordable digital tools and platforms like TikTok and SoundCloud have empowered countless bedroom producers to create and share music globally. However, achieving a polished, professional sound remains a significant challenge. Mixing and especially finalizing audio for clarity, loudness, and playback consistency remains a black box for many creators.

Most creators work with a final bounced mix, where all effects are flattened into a single stereo file, leaving no room for detailed post-production. Conventional post-production workflows often depend on access to individual stems and the expertise of trained engineers using specialized hardware or software. However, the majority of modern creators only have access to a single stereo mix—often recorded in non-ideal conditions—and lack the tools or knowledge to perform nuanced adjustments.

This thesis addresses a critical gap: **Can we design a system that restores and enhances fully mixed music audio, even when it is degraded and stemless, in a generalizable, accessible way?**

### Vision for Accessible Audio Intelligence

This work imagines a future where intelligent audio cleanup, handling things like echo, muddiness, or reverb, is available to any creator, regardless of environment or experience. Our system, which combines token representations, curriculum learning, and perceptually aligned objectives, aims to move beyond hand-crafted signal chains and towards learned audio enhancement built for realism and accessibility.

---

## Problem Statement

### Research Problem

**How can we restore and enhance fully mixed music audio - without access to stems, without expert supervision, and without manual DSP intervention - using a deep learning system that generalizes between genres, degradations, and real-world recording conditions?**

Fully mixed music tracks, or "bounced" files, embed not just the instruments and vocals but also every EQ curve, compression setting, reverb trail, and gain adjustment applied during mixing. Once exported, these effects are irreversibly entangled in the final waveform. Unlike speech, music is diverse in genre, instrumentation, tempo, and timbre, making the problem more difficult and much less studied when approached without source separation.

### Compounding Challenges

1. **No ground truth stems:** Most users have no access to isolated instrument or vocal tracks.
2. **Entangled effects:** Artifacts are non-linear and inseparable, invalidating additive noise models.
3. **Non-professional conditions:** Audio may be clipped, distorted, or captured on consumer-grade equipment.
4. **No definitive target:** Multiple "clean" versions may be equally perceptually valid, complicating supervision.

Most importantly, restoration must be **feasible**: it must operate on accessible hardware, provide interpretable outputs, and outperform existing approaches in perceptual quality and objective metrics.

### The Research Gap

While speech enhancement and stem-based mastering have received substantial attention, **there is no state-of-the-art system specifically designed to restore fully mixed, stemless music recordings under real-world conditions**. Most related work targets isolated effects or speech; none operate together in all five effects on degraded music without stems.

### Speech vs. Music: Fundamental Differences

![Speech vs Music Comparison](speech_vs_music.png)

*Visual comparison showing the fundamental differences between speech (narrowband, formant-based) and music (broadband, harmonic-rich) signals. This distinction is crucial for understanding why speech-focused models fail on music restoration tasks.*

---

## Mathematical Foundations

### Token-Based Representation

The core innovation of this work is operating entirely in **discrete token space** using Meta AI's EnCodec. Rather than predicting waveforms or spectrograms directly, our model learns to map degraded token sequences to clean token sequences.

#### Why Tokens?

1. **Compression and Speed:** A 30-second stereo clip is reduced to only a [16 × 2250] token matrix, allowing fast training and GPU-efficient batching.
2. **Semantic Abstraction:** Tokens encode perceptual features such as timbre, attack, stereo width, and distortion.
3. **Multi-Effect Compatibility:** EnCodec tokens reflect all aspects of musical structure, serving as a universal substrate for learning simultaneous restoration tasks.
4. **Phase Preservation:** Unlike spectrograms, tokens preserve phase information implicitly through the EnCodec decoder.

#### EnCodec Architecture

![EnCodec Architecture](encodec.png)

*EnCodec's encoder-quantizer-decoder architecture with residual vector quantization (RVQ) and adversarial discriminators for high-fidelity audio compression.*

#### EnCodec Configuration

- **Sample Rate:** 48,000 Hz (decoded), 22.05 kHz (training input)
- **Frame Rate:** 75 fps
- **Codebooks:** 16 RVQ stages
- **Codebook Size:** 1024 entries per codebook
- **Token Shape:** [16 × 2250] for 30-second clips
- **Total Bandwidth:** 24 kbps (stereo)

### EnCodec Tokenization Mathematics

EnCodec performs K-stage residual vector quantization to discretize continuous latent embeddings:

```
z_t ≈ Σ(k=1 to K) c_k[q_k(z_t)]
```

where `q_k(z_t)` is the index of the codebook (token) in stage k, and N is the number of centroids per codebook (typically N = 1024). This produces a token matrix:

```
tokens ∈ Z^(K × T)
```

The effective bit rate is:

```
B = (K · log₂(N) · r) / 1000
```

For our configuration: B = (16 · 10 · 75) / 1000 = 12.0 kbps per channel, or 24 kbps for stereo.

### Token U-Net Architecture Components

#### 1. Residual Blocks with GroupNorm and GELU

Each residual block implements:

```
y = x + F(x)
```

where F is a nonlinear transformation with GroupNorm (8 groups) and GELU activations:

```
F(x) = GELU(GN(Conv1D(Dropout(GELU(GN(Conv1D(x)))))))
```

#### 2. CBAM (Convolutional Block Attention Module)

CBAM applies sequential channel and temporal attention to enhance feature representations:

![CBAM Architecture](CBAM.png)

*Complete CBAM module showing channel attention (top) and spatial/temporal attention (bottom) pathways.*

**Channel Attention:**

```
z_c = GAP(x) = (1/T) Σ(t=1 to T) x_{:,t}
w_c = σ(W₂ · ReLU(W₁ · z_c)) ∈ (0,1)^C
```

**Temporal Attention:**

```
z_s = σ(Conv1D([Avg(x); Max(x)])) ∈ (0,1)^T
```

**Final Output:**

```
x' = x ⊙ w_c ⊙ z_s
```

![Spatial Attention Detail](spatial_attention.png)

*Detailed view of the spatial attention mechanism within CBAM, showing how max and average pooling are combined with convolution to generate spatial attention weights.*

#### 3. FiLM (Feature-wise Linear Modulation)

FiLM layers modulate features using learned affine parameters:

![FiLM Architecture](FILM.png)

*FiLM (Feature-wise Linear Modulation) layer showing how learned γ (gamma) and β (beta) parameters perform channel-wise scaling and shifting of feature maps.*

```
FiLM(x) = γ · x + β
```

where γ, β ∈ R^(C × 1) are learned affine parameters broadcast along the time axis.

#### 4. Learnable Scalar Skip Gates

To balance local detail and global abstraction, we use gated skip connections:

```
x^(i)_dec ← x^(i)_dec + σ(g_i) · Crop(x^(i)_enc)
```

Each g_i is a learned scalar parameter, passed through a sigmoid to constrain it to (0,1).

#### 5. Temporal Context Block (Dilated Convolution)

To support reverberant decay modeling and echo patterns, we include a dilated temporal context block:

```
x_context = Conv1D(x; k=9, dilation=2, padding=8)
```

This expands the effective receptive field without additional layers, enabling long-range dependency modeling for reverb tails and echo patterns.

### Loss Functions

#### Primary Token Cross-Entropy Loss

```
L_CE = (1/(B · n_q · T)) Σ(b=1 to B) Σ(q=1 to n_q) Σ(t=1 to T) CE(Z_{b,:,q,t}, Y_{b,q,t})
```

where Z ∈ R^(B × K × n_q × T) are the predicted logits and Y ∈ Z^(B × n_q × T) are the target tokens.

#### Auxiliary Losses

1. **Mask Head Loss (Dereverberation):**
   ```
   L_mask = (1/(B · n_q · T)) Σ(b,q,t) BCE(M̂_{b,q,t}, M_{b,q,t})
   ```

2. **Gain Head Loss:**
   ```
   L_gain = (1/B) Σ(b=1 to B) (g_b - g*_b)²
   ```

3. **Compression Head Loss:**
   ```
   L_comp = ||c - c*||²_2
   ```

4. **Stereo Head Loss:**
   ```
   L_stereo = ||s - s*||²_2
   ```

5. **Perceptual Head Loss:**
   ```
   L_perc = (1/B) Σ(b=1 to B) ||p̂_b - p_b||²_2
   ```

6. **Mel Spectrogram Loss:**
   ```
   L_mel = ||log(M̂ + ε) - log(M + ε)||₁
   ```

7. **STFT Spectral Loss:**
   ```
   L_STFT = (1/B) Σ(b=1 to B) |||STFT(ŷ_b)| - |STFT(y_b)|||²_2
   ```

#### Combined Loss Function

The total training loss combines all components:

```
L_total = L_CE 
        + λ_mask · L_mask
        + λ_perc · L_perc
        + λ_gain · L_gain
        + λ_stereo · L_stereo
        + λ_comp · L_comp
        + λ_STFT · L_STFT
        + λ_time · L_time
        + λ_mel · L_mel
        + λ_tok_spec · L_tok_spec
```

**Typical weight values:**
- λ_mask = λ_perc = λ_gain = λ_stereo = λ_comp = 0.1
- λ_STFT = 0.1, λ_time = 0.1
- λ_mel = 0.01, λ_tok_spec = 0.05

![Auxiliary Losses Over Training](auxiliary_weighted_losses.png)

*Progression of all auxiliary losses (mask, perceptual, gain, stereo, compression) throughout curriculum training. Note the stage-wise resets and overall decreasing trend.*

![All Weighted Losses](all_weighted_losses.png)

*Complete loss landscape showing primary cross-entropy loss alongside all weighted auxiliary components during training.*

### Audio Degradation Models

#### Equalization (EQ)

A parametric EQ filter is modeled as:

```
H_eq(f) = 1 + G / (1 + jQ(f/f₀ - f₀/f))
```

with parameters:
- Center frequency: f_c ~ U(300, 5000) Hz
- Quality factor: Q ~ U(0.5, 2.0)
- Gain: g ~ U(-6, +6) dB

#### Dynamic Range Compression

A soft-knee compressor:

```
y(t) = {
    x(t),                    if |x(t)| < θ
    θ + (|x(t)| - θ)/r,     otherwise
}
```

with parameters:
- Threshold: θ ~ U(-24, -6) dB
- Ratio: r ~ U(1.5, 4.0)
- Makeup gain: m ~ U(0, 3) dB

#### Reverb

Convolution with an exponentially decaying impulse response:

```
x_reverb(t) = (x * h_IR)(t)
```

where h_IR ~ exponential decay with:
- Decay constant: τ ~ U(0.2, 1.0)
- Impulse duration: T ~ U(50, 400) ms

#### Echo

Delayed and attenuated copy of the signal:

```
x_echo(t) = x(t) + α · x(t - τ)
```

with parameters:
- Delay: τ ~ U(100, 250) ms
- Attenuation: α ~ U(0.1, 0.5)

#### Gain Mismatch

Simple global amplitude scaling:

```
x̃[n] = 10^(g/20) · x[n],  g ~ U(-3, +3) dB
```

![Degradation Stack](degradation_stack.png)

*Illustration of how multiple degradations compound in real-world audio, creating entangled artifacts that require joint restoration.*

---

## Dataset and Preprocessing

### Dataset Selection: FMA Medium

We use the **Free Music Archive (FMA) Medium** dataset, which contains 25,000 tracks (30 seconds each) across 16 genres. This dataset is ideal because:

1. **Realistic Content:** Contains naturally produced or recorded bounced stereo tracks, often lacking professional mastering
2. **Diversity:** Spans genres, instrumentation, production styles, and loudness profiles
3. **Accessibility:** Creative Commons licensed, suitable for academic use
4. **Standardized Format:** Fixed 30-second segments at 22.05 kHz

### Curriculum Degradation Stages

Our training curriculum consists of five progressively difficult stages:

![Curriculum Stages](curriculum.png)

*Visualization of the curriculum learning progression, showing how degradations are introduced progressively from identity mappings to full random combinations.*

1. **Stage 0 – Identity:** Clean audio paired with itself; model must reconstruct tokens exactly
2. **Stage 1 – Single Effect:** One degradation (EQ, gain, compression, reverb, or echo)
3. **Stage 2 – Double Effects:** Two degradations applied in random order
4. **Stage 3 – Triple Effects:** Three randomly selected effects
5. **Stage 4 – Full Random:** Up to five simultaneous degradations

Stages 3 and 4 have "stronger" variants with widened parameter ranges to enforce generalization.

### Precomputed Token Bundles (.pt Files)

Each audio pair is preprocessed into a structured PyTorch `.pt` file containing:

- **X, Y:** Tokenized degraded and clean inputs ∈ Z^(n_q × T)
- **scales:** Per-frame scale factors from EnCodec ∈ R^T
- **Y_stft_mag:** STFT magnitudes of clean audio ∈ R^(n_q × F × T) (FP16)
- **mel_spec:** Mel spectrogram of clean waveform ∈ R^(M × T) (FP16)
- **metadata:** Sample rate, bandwidth, degradation parameters, SHA-256 hashes
- **wav_o, wav_m:** Cached original and degraded waveforms (optional)

![Token Bundle Structure](lumchbox.png)

*Visual representation of the "frozen lunchbox" .pt file format, showing all precomputed features bundled together for efficient training.*

This "frozen lunchbox" design enables:
- **40× speedup** in data loading compared to on-the-fly encoding
- Full reproducibility via SHA-256 hashes
- Support for all loss functions without recomputation

![Preprocessing Pipeline](structure.png)

*Complete pipeline from raw audio to precomputed token bundles, showing the demastering, tokenization, and feature extraction stages.*

### Preprocessing Pipeline

1. **Audio Loading:** Load 30-second clips, resample to 22.05 kHz, normalize to peak amplitude -1.0 dBFS
2. **Degradation Application:** Apply randomly sampled effects from the current curriculum stage
3. **EnCodec Tokenization:** Encode both clean and degraded audio to 48 kHz tokens at 24 kbps
4. **Feature Extraction:** Compute STFT magnitudes, mel spectrograms, and auxiliary statistics
5. **Serialization:** Save all features to `.pt` files with comprehensive metadata

---

## Training Pipeline

### Curriculum Learning Strategy

The training pipeline uses a **curriculum learning** approach where degradations are progressively introduced based on difficulty. This ensures:

1. **Stable Convergence:** Model learns simple restorations before complex ones
2. **Better Generalization:** Gradual exposure prevents overfitting to specific degradation patterns
3. **Interpretable Progress:** Each stage builds on previous knowledge

![Training Dynamics](stages_stuff.png)

*Training dynamics showing learning rate schedules, batch size adjustments, gradient accumulation, and recovery from OOM/NaN events throughout curriculum stages.*

### Stage Advancement Logic

Stage advancement is controlled by dual criteria:

1. **Validation Loss Plateau Detection:**
   - Exponential moving average: L̂_t^val = α L_t^val + (1 - α) L̂_{t-1}^val
   - Plateau detected when: |L̂_t^val - L̂_{t-k}^val| < δ for p consecutive epochs
   - Minimum epochs per stage: t_min = max(10, ⌊N_train / 100⌋)

2. **Training Loss Stagnation:**
   - Fallback criterion: σ(L_{t-n}^train, ..., L_t^train) < ε for n=8 epochs

### Training Configuration

#### Stage-Specific Hyperparameters

| Stage | Dropout | Bottleneck | Max LR | Epochs |
|-------|---------|------------|--------|--------|
| Stage 0 | 0.00 | False | 5e-4 | 15 (fixed) |
| Stage 1 | 0.00 | True | 4e-5 | Variable |
| Stage 1 Stronger | 0.00 | True | 3e-5 | Variable |
| Stage 2 | 0.05 | True | 2e-5 | Variable |
| Stage 3 | 0.08 | True | 1.5e-5 | Variable |
| Stage 3 Stronger | 0.10 | True | 1e-5 | Variable |
| Stage 4 | 0.15 | True | 8e-6 | Variable |
| Stage 4 Stronger | 0.15 | True | 5e-6 | Variable |

#### Optimization

- **Optimizer:** AdamW with β₁=0.9, β₂=0.999, weight decay 10⁻⁴
- **Learning Rate Schedule:** OneCycleLR for Stage 0, CosineAnnealingLR for later stages
- **Mixed Precision:** FP16 training with automatic mixed precision (AMP)
- **Gradient Clipping:** L2 norm clipping at threshold 10.0
- **Batch Size:** Adaptive, starting at 4, increasing up to 32 based on memory

#### Loss Weight Activation

Loss weights are activated progressively:
- **Stage 0:** Only L_CE
- **Stage 1+:** Audio-domain and auxiliary losses incrementally activated
- **Stage 3-4:** All loss terms enabled

### Training Infrastructure

#### Hardware

- **Primary:** NVIDIA RTX 6000 Ada (48 GB VRAM)
- **Development:** NVIDIA RTX 3070 Ti (8 GB VRAM)

#### Robustness Features

1. **OOM Recovery:** Automatic batch size reduction and checkpoint resumption
2. **NaN Detection:** Gradient monitoring and automatic recovery
3. **Checkpointing:** Every 5-10 epochs, plus best model tracking
4. **Audio Logging:** Periodic audio samples for perceptual validation
5. **Comprehensive Logging:** CSV logs, JSON metrics, visualization plots

---

## Evaluation and Results

### Quantitative Metrics

We evaluate restoration quality using multiple metrics:

1. **SNR (Signal-to-Noise Ratio):**
   ```
   SNR(x, x̂) = 10 · log₁₀(Σ_t x(t)² / Σ_t (x(t) - x̂(t))²)
   ```

2. **PESQ (Perceptual Evaluation of Speech Quality):** ITU-T P.862-based, range [-0.5, 4.5]

3. **STOI (Short-Time Objective Intelligibility):**
   ```
   STOI(x, x̂) = (1/K) Σ(k=1 to K) corr(x_k, x̂_k)
   ```

4. **ERLE (Echo Return Loss Enhancement):**
   ```
   ERLE(t) = 10 · log₁₀(E[y²(t)] / E[ê²(t)])
   ```

### Performance Summary

#### Comparison with Baselines

| Method | SNR ↑ | PESQ ↑ | STOI ↑ | Avg Score |
|--------|-------|--------|--------|-----------|
| **Token U-Net (Ours)** | **13.1** | **3.40** | **0.89** | **0.786** |
| DeepVQE | 14.1 | 3.41 | 0.91 | 0.775 |
| 2-Stage U-Net | 13.8 | 3.43 | 0.92 | 0.697 |
| Mimilakis DRC | 12.2 | 3.12 | - | 0.678 |

#### Per-Effect Performance

| Effect | Token U-Net | DeepVQE | Specialist Best |
|--------|-------------|---------|-----------------|
| EQ | 0.84 | 0.75 | 0.86 (Smit EQ) |
| Gain | 0.88 | 0.76 | - |
| Compression | 0.82 | 0.81 | 0.89 (Mimilakis) |
| Reverb | 0.80 | **0.85** | 0.91 (Speech Dereverb) |
| Echo | 0.83 | 0.84 | 0.89 (Li Echo) |

**Key Findings:**
- Token U-Net achieves balanced performance across all effects
- Specialist models excel on their target effect but underperform on others
- Token U-Net shows lowest variance (0.047) and highest robustness index (0.7883)

### Token Usage Analysis

![Token Histogram](histogram.png)

*Token usage histograms showing the distribution of EnCodec tokens before and after restoration. The model learns to utilize a broader token vocabulary, indicating improved representation capacity.*

### Qualitative Results

Listening tests on 30 tracks with mixed degradations revealed:

- **Increased transient clarity** compared to spectrogram-based U-Nets
- **Better stereo field preservation** and panning accuracy
- **Less "pumping" or spectral smearing** under heavy compression and reverb
- **More natural reverberation tails** when restoration fails (graceful degradation)

---

## Code Structure

### Core Modules (Curriculum_Tokenize_Master/)

```
Curriculum_Tokenize_Master/
├── token_unet.py          # Main Token U-Net architecture
├── token_train.py         # Curriculum-aware training script
├── token_dataset.py       # PyTorch dataset for token pairs
├── token_inference.py     # Inference and evaluation script
├── precompute_tokens.py   # Token precomputation pipeline
├── demastering.py         # Audio degradation generation
├── token_utils.py         # Utility functions
├── token_constants.py     # Constants and configuration
├── token_train_utils.py   # Training helper functions
├── token_data_utils.py    # Data loading utilities
├── token_plot_utils.py    # Visualization utilities
└── token_baseline.py      # Baseline model implementations
```

### Key Classes and Functions

#### `TokenUNet` (token_unet.py)

Main model class implementing the 1.08B parameter U-Net:

```python
class TokenUNet(nn.Module):
    def __init__(self, n_q: int, k: int = 1024, base_dim: int = 384, 
                 depth: int = 4, checkpointing: bool = False, 
                 use_bottleneck: bool = False, dropout: float = 0.10):
        # Architecture initialization
```

**Key Methods:**
- `forward(x)`: Forward pass returning logits and auxiliary outputs
- `set_dropout(dropout)`: Dynamically adjust dropout rate
- `get_num_params()`: Return parameter count

#### `TokenPairDataset` (token_dataset.py)

PyTorch Dataset for loading precomputed token pairs:

```python
class TokenPairDataset(Dataset):
    def __init__(self, base_dir, stages=None, model_type="48khz", 
                 bandwidth=24.0, force_audio=False, return_specs=False, 
                 return_meta=False, cache_wav=False):
        # Dataset initialization
```

**Key Features:**
- Loads from precomputed `.pt` files or encodes on-the-fly
- Supports multiple curriculum stages
- Returns tokens, scales, spectrograms, and metadata

#### Training Script (token_train.py)

Main training loop with curriculum learning:

```python
def train_curriculum(args):
    # Stage-by-stage training with automatic advancement
    # OOM recovery, NaN handling, checkpointing
    # Comprehensive logging and visualization
```

### Experimental Baselines

The repository includes several baseline implementations for comparison:

1. **Baseline Test/:** Early spectrogram-based U-Net experiments
2. **DeepUnet & LSTM src/:** Deep U-Net with LSTM parameter prediction
3. **CBAMFiLMUNet + InvLSTM src/:** U-Net with CBAM, FiLM, and inverse LSTM
4. **VocoderUNet & LSTM src/:** U-Net with neural vocoder integration
5. **GriffinLimNetTraining/:** Griffin-Lim based spectrogram inversion

---

## Usage Guide

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/takakhoo/AI_Neural_AudioCodec_Remastering.git
   cd AI_Neural_AudioCodec_Remastering
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install EnCodec:**
   ```bash
   cd externals/encodec
   pip install -e .
   cd ../..
   ```

### Data Preparation

1. **Download FMA Medium dataset:**
   ```bash
   # Place FMA Medium dataset in data/raw/fma_medium/fma_medium/
   ```

2. **Generate curriculum degradation stages:**
   ```bash
   # Stage 0 (Identity)
   python Curriculum_Tokenize_Master/demastering.py --stage 0 --seed 42
   
   # Stage 1 (Single effect)
   python Curriculum_Tokenize_Master/demastering.py --stage 1 --seed 42
   
   # Stage 3 with stronger parameters
   python Curriculum_Tokenize_Master/demastering.py --stage 3 --stronger --seed 42
   ```

3. **Precompute tokens:**
   ```bash
   python Curriculum_Tokenize_Master/precompute_tokens.py --stage stage0_identity
   python Curriculum_Tokenize_Master/precompute_tokens.py --stage stage1_single
   # ... repeat for all stages
   ```

### Training

**Full curriculum training:**
```bash
python Curriculum_Tokenize_Master/token_train.py \
    --base_dir experiments/curriculums \
    --output_dir CurriculumTraining \
    --resume_from_checkpoint path/to/checkpoint.pt  # Optional
```

**Train until specific stage:**
```bash
python Curriculum_Tokenize_Master/token_train.py \
    --base_dir experiments/curriculums \
    --output_dir CurriculumTraining \
    --until_stage stage3_triple
```

**Key training arguments:**
- `--base_dir`: Directory containing curriculum stage folders
- `--output_dir`: Output directory for checkpoints and logs
- `--resume_from_checkpoint`: Path to checkpoint for resuming
- `--until_stage`: Stop training at specified stage
- `--batch_size`: Initial batch size (default: 4)
- `--num_workers`: Data loading workers (default: 4)

### Inference

**Run inference on a stage:**
```bash
python Curriculum_Tokenize_Master/token_inference.py \
    --checkpoint path/to/best_model.pt \
    --stage_dir experiments/curriculums/stage4_full_stronger \
    --output_dir Inference_Results/stage4
```

**External audio inference:**
```bash
python external_inference.py \
    --checkpoint path/to/best_model.pt \
    --input_audio path/to/degraded_audio.wav \
    --output_audio path/to/restored_audio.wav
```

### Evaluation

The inference script automatically computes:
- SNR, PESQ, STOI metrics
- Mel spectrogram comparisons
- Audio output files
- CSV metrics files

---

## Future Work

### Genre-Aware Modeling

Incorporate genre awareness to align restorations with musical aesthetics:
- Explicit genre labels via FiLM conditioning
- Self-supervised genre inference from audio
- Genre-tuned attention mechanisms

### Stem-Aware Restoration

Extend to stem-level processing:
- Multi-source token encoders with inter-stem attention
- Source separation frontend (Conv-TasNet, Open-Unmix)
- Conditioned token diffusion for per-instrument enhancement

### User-Controlled Enhancement

Enable user-specified restoration goals:
- Text prompt conditioning (e.g., "make this warmer and less compressed")
- Reference audio conditioning
- Latent edit vectors for directional enhancement

### Real-Time Deployment

Optimize for live applications:
- Streaming-compatible architecture with chunk-wise processing
- Causal upsampling for low-latency inference
- Model compression via pruning, distillation, or quantization

### Prompt-Based Remixing

Advance toward zero-shot, prompt-driven systems:
- Multimodal token fusion (audio + text + genre tags)
- Semantic remixing (e.g., "make this more ambient")
- Unsupervised prompt tuning for label-free intent modeling

---

## Citation

If you use this work in your research, please cite:

```bibtex
@thesis{khoo2025tokenunet,
  title={Token U-Net: Neural Audio Codec Remastering for Full-Mix Music Restoration},
  author={Khoo, Taka},
  school={Dartmouth College},
  year={2025},
  type={Honors Thesis},
  advisor={Peter Chin},
  consultant={Michael Casey}
}
```

---

## Acknowledgments

- **Advisors:** Peter Chin (Primary), Michael Casey (Secondary)
- **Dartmouth LISP Lab** for computational resources
- **Meta AI** for the EnCodec codec
- **FMA Dataset** contributors for the training data

---

## License

This project is licensed under the MIT License. See LICENSE file for details.

---

## Contact

For questions, issues, or collaborations, please open an issue on GitHub or contact the author.

---

**Last Updated:** November 2025
