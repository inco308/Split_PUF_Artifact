"""图表生成: 论文 Fig 1-3 (S曲线 / 缩放律 / 架构无关性)

风格规范 (dataviz skill, 已验证调色板):
  - 6系列分类色: 蓝#2a78d6 橙#eb6834 青#1baf7a 黄#eda100 品红#e87ba4 绿#008300
  - 对比度WARN的颜色(青/黄/品红)必须有可见直接标签
  - 次级编码: 每系列不同marker形状(黑白打印/CVD场景)
  - 细线2px、浅网格、无衬线字体、文本用墨色token(不套系列色)
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, 'figures')
os.makedirs(OUT, exist_ok=True)

# 已验证的6色分类调色板 (adjacent-pairs PASS, light模式)
CAT = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300']
INK = '#0b0b0b'        # 主文本
INK2 = '#52514e'       # 次级文本
GRID = '#c9c8c4'       # 网格线(退后)
SURFACE = '#fcfcfb'    # 图面

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 9,
    'axes.edgecolor': INK2,
    'axes.labelcolor': INK,
    'xtick.color': INK2,
    'ytick.color': INK2,
    'axes.linewidth': 0.8,
    'figure.facecolor': SURFACE,
    'axes.facecolor': SURFACE,
})

S_CURVES = {
    '2-XOR': {'N': [10e3, 25e3, 50e3, 100e3], 'p': [0, 0.20, 0.30, 0.50], 'n': [10, 10, 10, 10]},
    '4-XOR': {'N': [75e3, 90e3, 100e3, 200e3], 'p': [0.80, 0.80, 0.90, 1.0], 'n': [10, 10, 10, 10]},
    '6-XOR': {'N': [200e3, 300e3, 400e3, 450e3, 500e3], 'p': [0.20, 0.33, 0.54, 0.71, 0.90], 'n': [5, 24, 24, 14, 10]},
    '7-XOR': {'N': [500e3, 750e3, 1e6, 1.25e6, 1.5e6, 2e6], 'p': [0, 0, 0.30, 0.80, 0.60, 1.0], 'n': [5, 5, 10, 10, 5, 5]},
    '8-XOR': {'N': [5e6, 5.5e6, 6e6, 8e6, 10e6], 'p': [0.28, 0.50, 0.27, 0.80, 0.64], 'n': [25, 14, 30, 5, 11]},
    '9-XOR': {'N': [5e6, 7.5e6, 10e6, 12e6, 15e6], 'p': [0.25, 0.40, 0.10, 0.0, 0.0], 'n': [8, 5, 10, 3, 3]},
}
MARKERS = ['o', 's', '^', 'D', 'v', 'P']


def wilson_ci(p, n):
    z = 1.96
    den = 1 + z ** 2 / n
    ctr = (p + z ** 2 / (2 * n)) / den
    m = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return np.maximum(0, ctr - m), np.minimum(1, ctr + m)


def clopper_pearson(p, n):
    from scipy.stats import beta
    lo = np.where(p == 0, 0.0, np.where(p == 1, 0.025 ** (1 / n),
                   beta.ppf(0.025, p * n, n - p * n + 1)))
    hi = np.where(p == 1, 1.0, np.where(p == 0, 1 - 0.025 ** (1 / n),
                   beta.ppf(0.975, p * n + 1, n - p * n)))
    return lo, hi


def ci_bounds(p, n):
    """与论文规则一致: n<=5或p∈{0,1}用CP, 否则Wilson"""
    p, n = np.asarray(p, float), np.asarray(n, int)
    lo, hi = np.zeros_like(p), np.ones_like(p)
    for i in range(len(p)):
        if n[i] <= 5 or p[i] == 0 or p[i] == 1:
            lo[i], hi[i] = clopper_pearson(p[i], n[i])
        else:
            lo[i], hi[i] = wilson_ci(p[i], n[i])
    return lo, hi


def fig_s_curves():
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    for i, (label, d) in enumerate(S_CURVES.items()):
        N = np.array(d['N']) / 1e6
        p = np.array(d['p'], float)
        n = np.array(d['n'], int)
        lo, hi = ci_bounds(p, n)
        c = CAT[i]
        ax.errorbar(N, p, yerr=[p - lo, hi - p], fmt='-', color=c,
                    linewidth=1.6, capsize=2, elinewidth=0.8, alpha=0.9,
                    zorder=2)
        # 次级编码: 每系列不同marker + 右端直接标签(对比度WARN补救)
        ax.plot(N, p, marker=MARKERS[i], linestyle='none', color=c,
                markersize=5, markeredgewidth=0.6, markeredgecolor=INK,
                zorder=3)
        ax.annotate(label, xy=(N[-1], p[-1]), xytext=(3, 0),
                    textcoords='offset points', fontsize=8, color=INK,
                    va='center')
    ax.axhline(0.5, color=GRID, linestyle=':', linewidth=1, zorder=1)
    ax.set_xscale('log')
    ax.set_xlim(6e-3, 45)
    ax.set_ylim(-0.06, 1.12)
    ax.set_xlabel('CRP budget $N$', fontsize=9)
    ax.set_ylabel('$P(\\mathrm{success} \\mid N, k)$', fontsize=9)
    ax.set_xticks([1e-2, 1e-1, 1, 10])
    ax.grid(True, axis='both', color=GRID, linewidth=0.5, alpha=0.5, zorder=0)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 's_curves.pdf'), dpi=150, bbox_inches='tight')
    plt.close(fig)


def fig_scaling_law():
    ks = np.array([2, 4, 6, 7, 8, 9])
    n50 = np.array([1.03e5, 7.5e4, 3.84e5, 1.32e6, 8.0e6, 12e6])
    labels = ['103k', '<75k', '384k', '1.32M', '8.0M', '>10M']
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    ax.semilogy(ks, n50, '-', color=CAT[0], linewidth=1.8, zorder=2)
    ax.plot(ks, n50, marker='o', linestyle='none', color=CAT[0],
            markersize=5, markeredgewidth=0.6, markeredgecolor=INK,
            markerfacecolor=SURFACE, zorder=3)
    for x, y, lab in zip(ks, n50, labels):
        va = 'bottom' if y < 1e6 else 'top'
        off = 8 if y < 1e6 else -8
        ax.annotate(lab, (x, y), xytext=(0, off), textcoords='offset points',
                    ha='center', va=va, fontsize=8, color=INK)
    # 每级倍数注释: 放在线段中点下方, 水平右移避开两端点标签
    for i in range(len(ks) - 1):
        ratio = n50[i + 1] / n50[i]
        mid_x = (ks[i] + ks[i + 1]) / 2
        mid_y = np.sqrt(n50[i] * n50[i + 1])
        ax.annotate(f'{ratio:.1f}$\\times$', (mid_x, mid_y), xytext=(8, -8),
                    textcoords='offset points', fontsize=7,
                    color=INK2, ha='center', va='top')
    ax.annotate('10-XOR: 15M OOM\non 24 GB', xy=(10, 2e7), fontsize=7,
                color=INK2, ha='center', style='italic')
    ax.set_xlim(1.5, 10.5)
    ax.set_xticks(range(2, 11))
    ax.set_xlabel('XOR complexity $k$', fontsize=9, labelpad=10)
    ax.set_ylabel('$N_{50}$ (CRPs)', fontsize=9, labelpad=6)
    ax.grid(True, axis='y', color=GRID, linewidth=0.5, alpha=0.5, zorder=0)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'scaling_law.pdf'), dpi=150, bbox_inches='tight')
    plt.close(fig)


def fig_architecture():
    names = ['Standard\nMLP', 'Deep\n6-layer', 'Wide\n$k{=}9$', 'Deep+\nWide',
             'ReLU+\nKaiming', 'Residual\n(148K)', 'No\nSigmoid', 'Big MLP\n(1.08M)']
    accs = [50.08, 50.05, 50.04, 49.96, 50.06, 50.08, 50.03, 50.05]
    fig, ax = plt.subplots(figsize=(5.4, 2.9))
    colors = [CAT[0]] + ['#d9d7d2'] * (len(accs) - 1)  # 基线强调, 其余中性
    bars = ax.bar(range(len(accs)), accs, color=colors, width=0.62, zorder=2)
    ax.axhline(99.71, color=CAT[0], linestyle='--', linewidth=1.2, zorder=3)
    ax.annotate('same model, 10M CRPs: 99.71%', xy=(len(accs) - 0.4, 99.9),
                fontsize=7, color=CAT[0], ha='right', va='bottom')
    ax.axhline(50, color=GRID, linestyle=':', linewidth=1, zorder=1)
    ax.annotate('random guessing: 50%', xy=(-0.4, 49.6),
                fontsize=7, color=INK2, ha='left', va='top')
    for i, (b, a) in enumerate(zip(bars, accs)):
        ax.annotate(f'{a:.2f}', (b.get_x() + b.get_width() / 2, a),
                    xytext=(0, 3), textcoords='offset points', ha='center',
                    fontsize=6.5, color=INK)
    ax.set_xticks(range(len(accs)))
    ax.set_xticklabels(names, fontsize=6.5, color=INK)
    ax.set_ylim(45, 104)
    ax.set_ylabel('test accuracy (%)', fontsize=9)
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'architecture_independence.pdf'), dpi=150,
                bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    fig_s_curves()
    fig_scaling_law()
    fig_architecture()
    print('figures written to', OUT)
