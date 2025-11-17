import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json

def plot_curves(logs, stage_name, stage_dir):
    """Generate comprehensive training visualization plots."""
    # Configure plot style
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 18
    })
    
    # 1. Main Training Overview (2x2)
    fig1, axes = plt.subplots(2, 2, figsize=(20, 16), constrained_layout=True)
    fig1.suptitle('Training Overview', fontsize=20)
    
    # Loss curves
    ax = axes[0,0]
    ax.plot(logs['train_loss'], label='Training', color='#1f77b4', linewidth=2)
    ax.plot(logs['val_loss'], label='Validation', color='#ff7f0e', linewidth=2)
    ax.set_title('Loss Curves')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cross Entropy Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Learning rate
    ax = axes[0,1]
    ax.plot(logs['lr'], color='#2ca02c', linewidth=2)
    ax.set_title('Learning Rate Schedule')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # Gradient norm
    ax = axes[1,0]
    ax.plot(logs['grad_norm'], color='#d62728', linewidth=2)
    ax.axhline(20, color='r', linestyle='--', alpha=0.5, label='Warning Threshold')
    ax.set_title('Gradient Norm')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Norm')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Stage progression
    ax = axes[1,1]
    ax.plot(logs['stage'], color='#9467bd', linewidth=2)
    ax.set_title('Stage Progression')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Stage Index')
    ax.grid(True, alpha=0.3)
    
    plt.savefig(stage_dir/'training_overview.png', dpi=300, bbox_inches='tight')
    plt.close(fig1)
    
    # 2. Detailed Loss Analysis
    fig2, axes = plt.subplots(2, 1, figsize=(20, 12), constrained_layout=True)
    fig2.suptitle('Detailed Loss Analysis', fontsize=20)
    
    # Loss with moving averages
    ax = axes[0]
    train_ma = moving_average(logs['train_loss'], window=3)
    val_ma = moving_average(logs['val_loss'], window=3)
    
    ax.plot(logs['train_loss'], label='Training (raw)', color='#1f77b4', alpha=0.3)
    ax.plot(train_ma, label='Training (3-epoch MA)', color='#1f77b4', linewidth=2)
    ax.plot(logs['val_loss'], label='Validation (raw)', color='#ff7f0e', alpha=0.3)
    ax.plot(val_ma, label='Validation (3-epoch MA)', color='#ff7f0e', linewidth=2)
    
    ax.set_title('Loss with Moving Averages')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cross Entropy Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Loss improvement rate
    ax = axes[1]
    train_improvement = [0] + [logs['train_loss'][i] - logs['train_loss'][i-1] 
                              for i in range(1, len(logs['train_loss']))]
    val_improvement = [0] + [logs['val_loss'][i] - logs['val_loss'][i-1] 
                            for i in range(1, len(logs['val_loss']))]
    
    ax.plot(train_improvement, label='Training', color='#1f77b4', linewidth=2)
    ax.plot(val_improvement, label='Validation', color='#ff7f0e', linewidth=2)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    ax.set_title('Per-Epoch Loss Improvement')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss Change')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.savefig(stage_dir/'loss_analysis.png', dpi=300, bbox_inches='tight')
    plt.close(fig2)
    
    # 3. Training Dynamics
    fig3, axes = plt.subplots(2, 1, figsize=(20, 12), constrained_layout=True)
    fig3.suptitle('Training Dynamics', fontsize=20)
    
    # Learning rate vs gradient norm
    ax = axes[0]
    ax2 = ax.twinx()
    ax.plot(logs['lr'], label='Learning Rate', color='#2ca02c', linewidth=2)
    ax2.plot(logs['grad_norm'], label='Gradient Norm', color='#d62728', linewidth=2)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate', color='#2ca02c')
    ax2.set_ylabel('Gradient Norm', color='#d62728')
    
    ax.tick_params(axis='y', labelcolor='#2ca02c')
    ax2.tick_params(axis='y', labelcolor='#d62728')
    
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # Stage transitions
    ax = axes[1]
    stage_changes = [i for i in range(1, len(logs['stage'])) 
                    if logs['stage'][i] != logs['stage'][i-1]]
    
    ax.plot(logs['stage'], color='#9467bd', linewidth=2)
    ax.scatter(stage_changes, [logs['stage'][i] for i in stage_changes], 
              color='red', s=100, zorder=5, label='Stage Transitions')
    
    ax.set_title('Stage Progression with Transitions')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Stage Index')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.savefig(stage_dir/'training_dynamics.png', dpi=300, bbox_inches='tight')
    plt.close(fig3)
    
    # 4. Audio metrics (if available)
    if 'pesq' in logs and 'stoi' in logs:
        fig4, axes = plt.subplots(2, 1, figsize=(20, 12), constrained_layout=True)
        fig4.suptitle('Audio Quality Metrics', fontsize=20)
        
        # PESQ
        ax = axes[0]
        ax.plot(logs['pesq'], color='#1f77b4', linewidth=2)
        ax.set_title('PESQ Score')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('PESQ')
        ax.grid(True, alpha=0.3)
        
        # STOI
        ax = axes[1]
        ax.plot(logs['stoi'], color='#ff7f0e', linewidth=2)
        ax.set_title('STOI Score')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('STOI')
        ax.grid(True, alpha=0.3)
        
        plt.savefig(stage_dir/'audio_metrics.png', dpi=300, bbox_inches='tight')
        plt.close(fig4)
    
    # Save raw metrics
    with open(stage_dir / 'training_metrics.json', 'w') as f:
        json.dump(logs, f, indent=2)

def moving_average(seq, window=3):
    """Compute moving average of sequence."""
    return np.convolve(seq, np.ones(window)/window, mode='valid')

def print_training_stats(logs):
    """Print comprehensive training statistics."""
    print("\nTraining Statistics:")
    print(f"Total epochs: {len(logs['train_loss'])}")
    print(f"Final train loss: {logs['train_loss'][-1]:.4f}")
    print(f"Final val loss: {logs['val_loss'][-1]:.4f}")
    print(f"Best val loss: {min(logs['val_loss']):.4f}")
    print(f"Final learning rate: {logs['lr'][-1]:.2e}")
    print(f"Final gradient norm: {logs['grad_norm'][-1]:.2f}")

def print_stage_transitions(logs):
    """Print stage transition information."""
    stages = logs['stage']
    transitions = [i for i in range(1, len(stages)) if stages[i] != stages[i-1]]
    
    print("\nStage Transitions:")
    for i in transitions:
        print(f"Epoch {i}: {stages[i-1]} -> {stages[i]}")
        print(f"  Train loss: {logs['train_loss'][i]:.4f}")
        print(f"  Val loss: {logs['val_loss'][i]:.4f}")

def print_batch_stats(logs):
    """Print batch size and gradient accumulation statistics."""
    print("\nBatch Statistics:")
    print(f"Initial batch size: {logs['batch_size'][0]}")
    print(f"Final batch size: {logs['batch_size'][-1]}")
    print(f"Initial grad accum: {logs['grad_accum'][0]}")
    print(f"Final grad accum: {logs['grad_accum'][-1]}")

def print_optimization_stats(logs):
    """Print optimization-related statistics."""
    print("\nOptimization Statistics:")
    print(f"Learning rate range: {min(logs['lr']):.2e} - {max(logs['lr']):.2e}")
    print(f"Gradient norm range: {min(logs['grad_norm']):.2f} - {max(logs['grad_norm']):.2f}")
    print(f"Number of OOM events: {sum(1 for x in logs.get('oom', []) if x)}")
    print(f"Number of NaN events: {sum(1 for x in logs.get('nan', []) if x)}")

def print_model_stats(model):
    """Print model architecture statistics."""
    print("\nModel Statistics:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")

def print_dataset_stats(ds):
    """Print dataset statistics."""
    print("\nDataset Statistics:")
    print(f"Total samples: {len(ds)}")
    print(f"Input shape: {ds[0][0].shape}")
    print(f"Target shape: {ds[0][1].shape}")
    if hasattr(ds, 'use_tokens'):
        print(f"Using tokens: {ds.use_tokens}")
    if hasattr(ds, 'n_q'):
        print(f"Number of codebooks: {ds.n_q}")