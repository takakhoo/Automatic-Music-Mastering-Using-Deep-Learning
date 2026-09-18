"""Small synthetic token-learning experiment, NOT an audio-restoration benchmark."""
import argparse
import json
import platform
from pathlib import Path
import torch
import torch.nn.functional as F
from token_unet import TokenUNet


def run(output, steps=200, seed=7, full_resolution_skip=False):
    if steps < 1:
        raise ValueError('steps must be positive')
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    config = dict(n_q=2, k=8, base_dim=16, depth=1, dropout=0., full_resolution_skip=full_resolution_skip)
    # Independent examples generated before training. No EnCodec/FMA download.
    train_target = torch.randint(8, (32, 2, 32))
    test_target = torch.randint(8, (16, 2, 32))
    train_input, test_input = (train_target + 1) % 8, (test_target + 1) % 8
    model = TokenUNet(**config)
    optimizer = torch.optim.Adam(model.parameters(), lr=.003)
    history = []
    def evaluate(x, y):
        model.eval()
        with torch.no_grad():
            logits = model(x)['logits']
            return {'cross_entropy': F.cross_entropy(logits, y).item(),
                    'accuracy': (logits.argmax(1) == y).float().mean().item()}
    before = evaluate(test_input, test_target)
    for step in range(steps):
        model.train()
        start = (step % 4) * 8
        logits = model(train_input[start:start+8])['logits']
        loss = F.cross_entropy(logits, train_target[start:start+8])
        if not torch.isfinite(loss):
            raise RuntimeError('non-finite training loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        if step == 0 or (step + 1) % 10 == 0 or step == steps-1:
            history.append({'step': step+1, 'batch_loss': loss.item(), **evaluate(test_input, test_target)})
    train, test = evaluate(train_input, train_target), evaluate(test_input, test_target)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / 'synthetic-demo.pt'
    torch.save({'config': config, 'state_dict': model.state_dict()}, checkpoint)
    loaded = torch.load(checkpoint, map_location='cpu', weights_only=True)
    reloaded = TokenUNet(**loaded['config']).eval()
    reloaded.load_state_dict(loaded['state_dict'])
    with torch.no_grad():
        original = model(test_input)['logits']
        restored = reloaded(test_input)['logits']
    torch.testing.assert_close(original, restored, rtol=0, atol=0)
    report = {'task': 'synthetic modular token shift reversal; NOT audio quality',
              'seed': seed, 'steps': steps, 'python': platform.python_version(), 'torch': str(torch.__version__),
              'config': config, 'parameters': sum(p.numel() for p in model.parameters()),
              'training_sequences': 32, 'held_out_sequences': 16, 'tokens_per_sequence': 64,
              'optimizer': {'name': 'Adam', 'learning_rate': .003, 'batch_size': 8, 'gradient_clip_norm': 5.},
              'copy_input_accuracy': 0., 'uniform_chance_accuracy': .125,
              'before_training': before, 'after_training': {'train': train, 'held_out': test},
              'checkpoint_roundtrip_max_error': (original-restored).abs().max().item(), 'history': history}
    (output / 'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), constrained_layout=True)
    axes[0].plot([0]+[r['step'] for r in history], [before['cross_entropy']]+[r['cross_entropy'] for r in history], color='#3b6c9b')
    axes[0].set(title='Held-out token cross-entropy', xlabel='Updates', ylabel='Cross-entropy')
    axes[1].plot([0]+[r['step'] for r in history], [before['accuracy']]+[r['accuracy'] for r in history], color='#21816f', label='Reduced Token U-Net')
    axes[1].axhline(.125, color='#ad7040', linestyle='--', label='Uniform chance')
    axes[1].axhline(0, color='#888888', linestyle=':', label='Copy degraded input')
    axes[1].set(title='Synthetic task, not restored music', xlabel='Updates', ylabel='Held-out accuracy', ylim=(-.02, 1.02))
    axes[1].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.grid(alpha=.15)
    fig.savefig(output/'learning-curve.png', dpi=180)
    plt.close(fig)
    print(json.dumps({k: report[k] for k in ['parameters', 'before_training', 'after_training', 'checkpoint_roundtrip_max_error']}, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='results/cpu-token-learning')
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--full-resolution-skip', action='store_true')
    args = parser.parse_args()
    run(args.output, args.steps, args.seed, args.full_resolution_skip)
