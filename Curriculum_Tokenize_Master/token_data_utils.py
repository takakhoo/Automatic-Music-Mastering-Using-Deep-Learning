import torch
from torch.utils.data import DataLoader, random_split
from token_dataset import TokenPairDataset, pad_collate
from pathlib import Path

# Constants
VAL_FRAC = 0.10
TEST_FRAC = 0.10
STAGE0_MIN_EPOCHS = 40
STAGE1_MIN_EPOCHS = 150

def create_loaders(train_ds, val_ds, test_ds, batch_size, dev):
    """Create data loaders with appropriate settings."""
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                  collate_fn=pad_collate, num_workers=4, 
                  pin_memory=(dev.type=="cuda")),
        DataLoader(val_ds, batch_size=batch_size*2, shuffle=False, 
                  collate_fn=pad_collate, num_workers=2,
                  pin_memory=(dev.type=="cuda")),
        DataLoader(test_ds, batch_size=batch_size*2, shuffle=False, 
                  collate_fn=pad_collate, num_workers=2,
                  pin_memory=(dev.type=="cuda"))
    )

def get_epochs_per_stage(n_train):
    """Get maximum epochs per stage, with special handling for early stages."""
    if n_train < 2000:
        return 200  # More epochs for small datasets
    elif n_train < 10000:
        return 150  # Increased from 100
    else:
        return 100  # Increased from 50

def get_min_epochs_per_stage(n_train, stage_name):
    """Get minimum epochs per stage, with special handling for early stages."""
    if stage_name == "stage0_identity":
        return STAGE0_MIN_EPOCHS
    elif stage_name == "stage1_single":
        return STAGE1_MIN_EPOCHS
    elif n_train < 2000:
        return 50
    return max(20, n_train // 100)

def verify_train_val_split(ds: TokenPairDataset,
                          stage_name: str,
                          stage_dir: Path) -> None:
    """Verify train/val/test splits are valid.
    
    Args:
        ds: TokenPairDataset to verify
        stage_name: Current stage name
        stage_dir: Directory to save verification results
    """
    print(f"\nVerifying data splits for {stage_name}...")
    
    # Check if using tokens or audio
    if hasattr(ds, 'token_files') and ds.token_files:
        print("Using tokens")
    else:
        print("Using audio")
    
    # Print total dataset size
    print(f"Total dataset size: {len(ds)}")
    
    # Get split sizes
    n_val = max(1, int(len(ds) * VAL_FRAC))
    n_test = max(1, int(len(ds) * TEST_FRAC))
    n_train = len(ds) - n_val - n_test
    
    print("\nSplit sizes:")
    print(f"Train: {n_train} samples")
    print(f"Val: {n_val} samples")
    print(f"Test: {n_test} samples")
    
    # Get vocabulary size per codebook (1024 for EnCodec)
    vocab_size = 1024
    
    # Verify first few samples
    for i in range(min(3, len(ds))):
        # Unpack only the first 3 values (X, Y, Y_scales)
        X, Y, Y_scales = ds[i][:3]
        print(f"[{i}] {ds.token_files[i].name if hasattr(ds, 'token_files') else 'audio'} → "
              f"X {tuple(X.shape)} Y {tuple(Y.shape)}")
        
        # Verify shapes
        assert X.shape[0] == ds.n_q, f"Expected {ds.n_q} codebooks, got {X.shape[0]}"
        assert Y.shape[0] == ds.n_q, f"Expected {ds.n_q} codebooks, got {Y.shape[0]}"
        assert X.shape[1] == Y.shape[1], f"Token length mismatch: {X.shape[1]} vs {Y.shape[1]}"
        assert Y_scales.shape[0] == Y.shape[1], f"Scale length mismatch: {Y_scales.shape[0]} vs {Y.shape[1]}"
        
        # Verify value ranges
        assert X.min() >= 0 and X.max() < vocab_size, f"X tokens out of range: [{X.min()}, {X.max()}]"
        assert Y.min() >= 0 and Y.max() < vocab_size, f"Y tokens out of range: [{Y.min()}, {Y.max()}]"
        assert Y_scales.min() > 0, f"Found zero scales: min={Y_scales.min()}"
    
    print("\nData split verification complete!")

def flatten_logits_targets(logits: torch.Tensor, targets: torch.Tensor, n_q: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Flatten logits and targets for loss computation.
    
    Args:
        logits: Shape (batch, n_q, seq_len, vocab_size)
        targets: Shape (batch, n_q, seq_len)
        n_q: Number of codebooks
        
    Returns:
        tuple of (flattened_logits, flattened_targets)
    """
    # Reshape logits to (batch * n_q * seq_len, vocab_size)
    batch_size = logits.shape[0]
    seq_len = logits.shape[2]
    flattened_logits = logits.reshape(-1, logits.shape[-1])
    
    # Reshape targets to (batch * n_q * seq_len)
    flattened_targets = targets.reshape(-1)
    
    return flattened_logits, flattened_targets

def print_dataset_stats(ds, stage_name):
    """Print dataset statistics."""
    print(f"\nDataset Statistics for {stage_name}:")
    print(f"Total samples: {len(ds)}")
    print(f"Input shape: {ds[0][0].shape}")
    print(f"Target shape: {ds[0][1].shape}")
    if hasattr(ds, 'use_tokens'):
        print(f"Using tokens: {ds.use_tokens}")
    if hasattr(ds, 'n_q'):
        print(f"Number of codebooks: {ds.n_q}") 