"""分析工具: 表1置信区间 + N50 logistic拟合

对应论文 Table 1 (Wilson/Clopper-Pearson CI) 和 §4 的 N50 拟合。
"""
import csv
import os
import sys
from math import sqrt

import numpy as np
from scipy.optimize import curve_fit

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data')

# (k, N, 来源CSV列表, 与论文Table 1的聚合方式一致)
TABLE1_SOURCES = {
    '2-XOR': [(10_000, ['2xor_baseline.csv']), (25_000, ['2xor_baseline.csv']),
              (50_000, ['2xor_baseline.csv']), (100_000, ['2xor_baseline.csv'])],
    '4-XOR': [(75_000, ['4xor_N50.csv']), (90_000, ['4xor_N50.csv']),
              (100_000, ['next_experiments.csv']), (200_000, ['next_experiments.csv'])],
    '6-XOR': [(200_000, ['6xor_200k.csv']),
              (300_000, ['baseline_mlp_lr.csv', 'phase7_variance.csv', 'phase8_bootstrap.csv']),
              (400_000, ['baseline_mlp_lr.csv', 'phase7_variance.csv', 'phase8_bootstrap.csv']),
              (450_000, ['next_experiments.csv', 'phase7_variance.csv']),
              (500_000, ['baseline_mlp_lr.csv'])],
    '7-XOR': [(500_000, ['7xor_baseline.csv']),
              (1_000_000, ['7xor_baseline.csv', '7xor_N50_fine.csv']),
              (1_250_000, ['7xor_N50_fine.csv']),
              (2_000_000, ['7xor_baseline.csv'])],
    '8-XOR': [(5_000_000, ['phase7_variance.csv', 'phase8_bootstrap.csv',
                           'baseline_mlp_lr.csv', '8xor_v2_results.csv']),
              (5_500_000, ['baseline_mlp_lr.csv', 'phase7_variance.csv']),
              (6_000_000, ['next_experiments.csv', '8xor_6M_replication.csv']),
              (8_000_000, ['8xor_8M.csv']),
              (10_000_000, ['8xor_10M_replication.csv', '8xor_v2_results.csv'])],
    '9-XOR': [(5_000_000, ['9xor_test.csv']), (7_500_000, ['9xor_7_5M.csv']),
              (10_000_000, ['9xor_10M.csv']), (12_000_000, ['9xor_12M.csv']),
              (15_000_000, ['9xor_15M.csv'])],
}


def load_ok_rows(csv_name, data_val=None):
    """加载CSV的成功标记(0/1), 按data值过滤。兼容三种schema:
    - ok列 (OOM行剔除)
    - mlp_acc列 (无ok列, 按>90%推导)
    - success列 + data_size列 (8xor_v2_results.csv)"""
    path = os.path.join(DATA, csv_name)
    rows = list(csv.DictReader(open(path)))
    out = []
    for r in rows:
        dkey = 'data' if 'data' in r else 'data_size'
        if data_val is not None and int(float(r[dkey])) != data_val:
            continue
        if 'ok' in r and r['ok'] != '':
            if r.get('acc') in ('OOM', ''):
                continue
            out.append(int(r['ok']))
        elif 'mlp_acc' in r and r['mlp_acc'] != '':
            out.append(1 if float(r['mlp_acc'].rstrip('%')) > 90 else 0)
        elif 'success' in r and r['success'] != '':
            out.append(int(r['success']))
    return out


def wilson(x, n):
    if n == 0:
        return 0.0, 1.0
    p = x / n
    z = 1.96
    den = 1 + z ** 2 / n
    ctr = (p + z ** 2 / (2 * n)) / den
    m = z * sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return max(0.0, ctr - m), min(1.0, ctr + m)


def clopper_pearson(x, n):
    from scipy.stats import beta
    if x == 0:
        return 0.0, 1 - 0.025 ** (1 / n)
    if x == n:
        return 0.025 ** (1 / n), 1.0
    return beta.ppf(0.025, x, n - x + 1), beta.ppf(0.975, x + 1, n - x)


def table1():
    print('=== Table 1 复算 ===')
    for k, points in TABLE1_SOURCES.items():
        for entry in points:
            N, srcs = entry[0], entry[1]
            oks = []
            for src in srcs:
                oks += load_ok_rows(src, N)
            n = len(oks)
            if n == 0:
                print(f'{k:>6} {N:>12,}: 无数据 (检查 {srcs})')
                continue
            x = sum(oks)
            p = x / n
            if n <= 5 or x == 0 or x == n:
                lo, hi = clopper_pearson(x, n)
            else:
                lo, hi = wilson(x, n)
            print(f'{k:>6} {N:>12,}: {x}/{n} = {p*100:.0f}%  CI [{lo*100:.0f}%, {hi*100:.0f}%]')


def n50_fit():
    print('\n=== N50 logistic拟合 ===')
    data = {}
    for k, points in TABLE1_SOURCES.items():
        xs, ps = [], []
        for entry in points:
            N, srcs = entry[0], entry[1]
            oks = []
            for src in srcs:
                oks += load_ok_rows(src, N)
            if oks:
                xs.append(N)
                ps.append(sum(oks) / len(oks))
        data[k] = (np.array(xs, dtype=float), np.array(ps))

    def logistic_logN(N, logN50, w):
        return 1 / (1 + np.exp(-(np.log10(N) - logN50) / w))

    for k, (Ns, ps) in data.items():
        mask = (ps > 0.01) & (ps < 0.99)
        if mask.sum() < 2:
            print(f'{k}: 内部点不足, 无法拟合')
            continue
        popt, pcov = curve_fit(logistic_logN, Ns[mask], ps[mask],
                               p0=[np.log10(np.median(Ns[mask])), 0.3],
                               bounds=([3, 0.05], [8, 2.0]), maxfev=10000)
        N50 = 10 ** popt[0]
        err = N50 * np.log(10) * sqrt(pcov[0, 0])
        pred = logistic_logN(Ns, *popt)
        ss_res = np.sum((ps - pred) ** 2)
        ss_tot = np.sum((ps - ps.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        print(f'{k}: N50 = {N50:.3e} +/- {err:.3e}  R2 = {r2:.3f}')


if __name__ == '__main__':
    table1()
    n50_fit()
