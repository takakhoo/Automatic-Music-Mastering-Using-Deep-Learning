"""Paired, three-seed ablation of the optional full-resolution token skip."""
import argparse
import json
from pathlib import Path
from reproduce_cpu import run


def compare(output):
    output = Path(output)
    rows = []
    for seed in (7, 19, 41):
        for enabled in (False, True):
            label = 'full-resolution-skip' if enabled else 'original'
            result = run(output / f'seed-{seed}-{label}', seed=seed, full_resolution_skip=enabled)
            rows.append({'seed': seed, 'variant': label, 'parameters': result['parameters'],
                         'train_accuracy': result['after_training']['train']['accuracy'],
                         'held_out_accuracy': result['after_training']['held_out']['accuracy'],
                         'held_out_cross_entropy': result['after_training']['held_out']['cross_entropy'],
                         'checkpoint_max_error': result['checkpoint_roundtrip_max_error']})
    (output / 'comparison.json').write_text(json.dumps(rows, indent=2, allow_nan=False)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 3.7), constrained_layout=True)
    for index, seed in enumerate((7, 19, 41)):
        values = [r['held_out_accuracy']*100 for r in rows if r['seed'] == seed]
        ax.plot([0, 1], values, 'o-', label=f'Seed {seed}', linewidth=1.5)
    ax.axhline(12.5, linestyle='--', color='#888888', linewidth=1, label='Uniform chance')
    ax.set(xticks=[0, 1], xticklabels=['Original architecture', 'Full-resolution skip'],
           ylabel='Held-out token accuracy (%)', ylim=(0, 100),
           title='Synthetic token-shift task: paired architecture ablation')
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis='y', alpha=.15)
    fig.savefig(output/'comparison.png', dpi=180)
    plt.close(fig)
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='results/cpu-ablation')
    args = parser.parse_args()
    compare(args.output)
