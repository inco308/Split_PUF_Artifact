"""图表生成: 论文 Fig 1-3 (S曲线 / 缩放律 / 架构无关性)"""
import csv
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data')
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'figures')
os.makedirs(OUT, exist_ok=True)

S_CURVES = {
    '2-XOR': {'N': [10e3, 25e3, 50e3, 100e3], 'p': [0, 0.20, 0.30, 0.50], 'n': [10, 10, 10, 10]},
    '4-XOR': {'N': [75e3, 90e3, 100e3, 200e3], 'p': [0.80, 0.80, 0.90, 1.0], 'n': [10, 10, 10, 10]},
    '6-XOR': {'N': [200e3, 300e3, 400e3, 450e3, 500e3], 'p': [0.20, 0.33, 0.54, 0.71, 0.90], 'n': [5, 24, 24, 14, 10]},
    '7-XOR': {'N': [500e3, 750e3, 1e6, 1.25e6, 1.5e6, 2e6], 'p': [0, 0, 0.30, 0.80, 0.60, 1.0], 'n': [5, 5, 10, 10, 5, 5]},
    '8-XOR': {'N': [5e6, 5.5e6, 6e6, 8e6, 10e6], 'p': [0.28, 0.50, 0.27, 0.80, 0.64], 'n': [25, 14, 30, 5, 11]},
    '9-XOR': {'N': [5e6, 7.5e6, 10e6, 12e6, 15e6], 'p': [0.25, 0.40, 0.10, 0.0, 0.0], 'n': [8, 5, 10, 3, 3]},
}


def wilson_ci(p, n):
    z = 1.96
    den = 1 + z ** 2 / n
    ctr = (p + z ** 2 / (2 * n)) / den
    m = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return np.maximum(0, ctr - m), np.minimum(1, ctr + m)


def fig_s_curves():
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.viridis(np.linspace(0, 1, len(S_CURVES)))
    for (label, d), c in zip(S_CURVES.items(), colors):
        N = np.array(d['N']) / 1e6
        p = np.array(d['p'])
        n = np.array(d['n'])
        lo, hi = wilson_ci(p, n)
        ax.errorbar(N, p, yerr=[p - lo, hi - p], fmt='o-', color=c, label=label,
                    capsize=3, markersize=6, linewidth=1.5)
    ax.set_xscale('log')
    ax.set_xlabel('Number of CRPs (N)', fontsize=12)
    ax.set_ylabel('P(success | N, k)', fontsize=12)
    ax.set_title('Probabilistic Phase Transition in ML Attacks on XOR APUFs', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9, ncol=2)
    ax.set_ylim(-0.05, 1.1)
    ax.set_xlim(5e-3, 30)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 's_curves.pdf'), dpi=150, bbox_inches='tight')
    plt.close()


def fig_scaling_law():
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ks = np.array([2, 4, 6, 7, 8, 9])
    n50 = np.array([1.03e5, 7.5e4, 3.84e5, 1.32e6, 8.0e6, 12e6])
    labels = ['103k', '<75k', '384k', '1.32M', '8.0M', '>10M']
    ax.semilogy(ks, n50, 'o-', color='#2196F3', markersize=10, linewidth=2,
                markerfacecolor='white', markeredgewidth=2)
    for i, lab in enumerate(labels):
        ax.annotate(lab, (ks[i], n50[i]), textcoords='offset points', xytext=(0, 12), ha='center', fontsize=8)
    ax.annotate('10-XOR: OOM\nat 15M on 24GB', xy=(10, 2e7), fontsize=8, color='red', ha='center', style='italic')
    ax.set_xlabel('XOR complexity (k)', fontsize=12)
    ax.set_ylabel('N50 (CRPs for 50% success)', fontsize=12)
    ax.set_title('Scaling of Attack Difficulty with XOR Complexity', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(2, 11))
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'scaling_law.pdf'), dpi=150, bbox_inches='tight')
    plt.close()


def fig_architecture():
    archs = ['Standard\nMLP\n(k=8)', 'Deep\n6-layer\n(k=8)', 'Wide\n(k=9)', 'Deep+\nWide\n(k=9,6L)',
             'ReLU+\nKaiming', 'Residual\n(148K)', 'No\nSigmoid', 'Big MLP\n(1.08M)']
    accs = [50.08, 50.05, 50.04, 49.96, 50.06, 50.08, 50.03, 50.05]
    fig, ax = plt.subplots(figsize=(8, 3.5))
    bars = ax.bar(range(len(archs)), accs, color=['#607D8B'] * 8)
    bars[0].set_color('#4CAF50')
    ax.axhline(y=99.71, color='#4CAF50', linestyle='--', linewidth=2, label='Same model with 10M data: 99.71%')
    ax.axhline(y=50, color='red', linestyle=':', alpha=0.3, label='Random guessing: 50%')
    ax.set_xticks(range(len(archs)))
    ax.set_xticklabels(archs, fontsize=7)
    ax.set_ylabel('Test Accuracy (%)', fontsize=11)
    ax.set_title('Architecture Independence: 8-XOR with 2M CRPs', fontsize=12, fontweight='bold')
    ax.set_ylim(45, 105)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'architecture_independence.pdf'), dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    fig_s_curves()
    fig_scaling_law()
    fig_architecture()
    print('figures written to', OUT)
