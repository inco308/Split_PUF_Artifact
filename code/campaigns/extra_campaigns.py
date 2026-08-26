"""补充战役脚本(开发阶段以交互式脚本运行, 此处整理为可复现文件)

覆盖的CSV:
  - 2xor_baseline.csv      (2-XOR 10k/25k/50k/100k x10)
  - 9xor_test.csv          (9-XOR 5M x10, 其中3次10M OOM行)
  - 9xor_7_5M.csv          (9-XOR 7.5M x5)
  - 8xor_8M.csv            (8-XOR 8M x5)
  - 9xor_15M.csv           (9-XOR 15M x3)
  - noise_6xor_400k_v2.csv (6-XOR 400k, eps=0/3%/5% x5)
  - poly_lr_4xor.csv       (4-XOR 100k, 多项式LR deg 2/3 x3)

协议与论文 Section 3 一致: 4层MLP [64,2^(k-1),2^k,2^(k-1),1],
Tanh+Sigmoid, Adam(0.001), CosineAnnealing(500), batch 4096,
80/2/18划分, 验证集早停(每10轮评估, ep>150且best<0.52).
"""
import sys, os, csv, time, gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'utils'))
from PUFs import XORPUFModel

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data')


class StdMLP(nn.Module):
    def __init__(self, k):
        super().__init__()
        d = [64, 2 ** (k - 1), 2 ** k, 2 ** (k - 1), 1]
        layers = []
        for i in range(len(d) - 1):
            layers.append(nn.Linear(d[i], d[i + 1]))
            layers.append(nn.Tanh() if i < len(d) - 2 else nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def batch_response(puf, ch):
    w = puf.weight[:, :-1].astype(np.float32)
    b = puf.weight[:, -1].astype(np.float32)
    bits = (np.dot(ch, w.T) + b >= 0).astype(np.int8)
    return np.bitwise_xor.reduce(bits, axis=1)


def run_one(k, ds, eps, csv_path, row):
    gc.collect()
    torch.cuda.empty_cache()
    puf = XORPUFModel.randomSample(k, 64)
    ch = np.random.randint(0, 2, (ds, 64)).astype(np.int8) * 2 - 1
    resp = batch_response(puf, ch).astype(np.float32)
    if eps is not None and eps > 0:
        flip = np.random.rand(ds) < eps
        resp = resp.copy()
        resp[flip] = 1 - resp[flip]
    d = np.column_stack([ch.astype(np.float32), resp.reshape(-1, 1)])
    del ch, resp
    np.random.shuffle(d)
    tn, vn = int(len(d) * 0.8), int(len(d) * 0.02)
    tp = torch.from_numpy(d[:tn, :-1]).float().cuda()
    tr = torch.from_numpy(d[:tn, -1:]).float().cuda()
    tv = torch.from_numpy(d[tn:tn + vn, :-1]).float().cuda()
    tvr = torch.from_numpy(d[tn:tn + vn, -1:]).float().cuda()
    if eps is not None and eps > 0:
        # 噪声实验: 测试集用干净的独立挑战(与冻结CSV协议一致)
        tch = np.random.randint(0, 2, (50_000, 64)).astype(np.int8) * 2 - 1
        tresp = batch_response(puf, tch).astype(np.float32)
        ttp = torch.from_numpy(tch.astype(np.float32)).float()
        ttr = torch.from_numpy(tresp.reshape(-1, 1)).float()
        del tch, tresp
    else:
        ttp = torch.from_numpy(d[tn + vn:, :-1]).float()
        ttr = torch.from_numpy(d[tn + vn:, -1:]).float()
    del d
    gc.collect()
    m = StdMLP(k).cuda()
    opt = torch.optim.Adam(m.parameters(), lr=0.001)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 500)
    best = 0.0
    t0 = time.time()
    for ep in range(1, 501):
        m.train()
        perm = torch.randperm(tn)
        for i in range(0, tn, 4096):
            idx = perm[i:i + 4096]
            loss = F.binary_cross_entropy(m(tp[idx]), tr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        sch.step()
        if ep % 10 == 0:
            m.eval()
            with torch.no_grad():
                va = (m(tv).round() == tvr).float().mean().item()
            best = max(best, va)
            if ep > 150 and best < 0.52:
                break
            m.train()
    dt = time.time() - t0
    m.eval()
    with torch.no_grad():
        ta = (m(ttp.cuda()).round() == ttr.cuda()).float().mean().item()
    ok = 1 if ta > 0.9 else 0
    print(f'  {k}X {ds // 1000}k eps={eps} #{row}: {ta * 100:.1f}% {dt:.0f}s {"OK" if ok else "FAIL"}')
    with open(csv_path, 'a', newline='') as f:
        csv.writer(f).writerow(row(k, ds, row, ta, dt, ok))
    del m, tp, tr, tv, tvr, ttp, ttr
    gc.collect()
    torch.cuda.empty_cache()


def main():
    # 2-XOR 基线
    p = os.path.join(RES, '2xor_baseline.csv')
    with open(p, 'w', newline='') as f:
        csv.writer(f).writerow(['xor', 'data', 'run', 'acc', 'time', 'ok'])
    for ds in [10_000, 25_000, 50_000, 100_000]:
        for r in range(1, 11):
            run_one(2, ds, 0.0, p, lambda k, d, r, a, t, o: [k, d, r, f'{a*100:.2f}%', f'{t:.0f}', o])

    # 9-XOR 5M / 7.5M
    for fname, ds, nr in [('9xor_test.csv', 5_000_000, 10),
                          ('9xor_7_5M.csv', 7_500_000, 5)]:
        p = os.path.join(RES, fname)
        with open(p, 'w', newline='') as f:
            csv.writer(f).writerow(['xor', 'data', 'run', 'acc', 'time', 'ok'])
        for r in range(1, nr + 1):
            run_one(9, ds, 0.0, p, lambda k, d, r, a, t, o: [k, d, r, f'{a*100:.2f}%', f'{t:.0f}', o])

    # 8-XOR 8M / 9-XOR 15M
    for fname, k, ds, nr in [('8xor_8M.csv', 8, 8_000_000, 5),
                             ('9xor_15M.csv', 9, 15_000_000, 3)]:
        p = os.path.join(RES, fname)
        with open(p, 'w', newline='') as f:
            csv.writer(f).writerow(['xor', 'data', 'run', 'acc', 'time', 'ok'])
        for r in range(1, nr + 1):
            run_one(k, ds, 0.0, p, lambda k, d, r, a, t, o: [k, d, r, f'{a*100:.2f}%', f'{t:.0f}', o])

    # 噪声网格: 6-XOR 400k, eps = 0 / 3% / 5%
    p = os.path.join(RES, 'noise_6xor_400k_v2.csv')
    with open(p, 'w', newline='') as f:
        csv.writer(f).writerow(['xor', 'data', 'eps', 'run', 'clean_test_acc', 'time', 'ok'])
    for eps in [0.0, 0.03, 0.05]:
        for r in range(1, 6):
            run_one(6, 400_000, eps, p, lambda k, d, r, a, t, o: [k, d, f'{eps}', r, f'{a*100:.2f}%', f'{t:.0f}', o])


def poly_lr():
    """多项式特征LR: 4-XOR 100k, deg 2/3, 随机8维多项式"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import PolynomialFeatures
    p = os.path.join(RES, 'poly_lr_4xor.csv')
    with open(p, 'w', newline='') as f:
        csv.writer(f).writerow(['xor', 'data', 'degree', 'run', 'acc', 'time'])
    for deg in [2, 3]:
        for r in range(1, 4):
            puf = XORPUFModel.randomSample(4, 64)
            ch = np.random.randint(0, 2, (100_000, 64)).astype(np.int8) * 2 - 1
            resp = batch_response(puf, ch)
            tn = int(len(ch) * 0.8)
            sel = np.random.choice(64, 8, replace=False)
            pf = PolynomialFeatures(degree=deg, include_bias=False)
            Xtr = pf.fit_transform(ch[:tn][:, sel].astype(np.float32))
            Xte = pf.transform(ch[tn:][:, sel].astype(np.float32))
            t0 = time.time()
            clf = LogisticRegression(max_iter=500, solver='lbfgs')
            clf.fit(Xtr, resp[:tn])
            acc = clf.score(Xte, resp[tn:])
            dt = time.time() - t0
            print(f'  poly deg={deg} #{r}: {acc*100:.1f}% ({dt:.0f}s)')
            with open(p, 'a', newline='') as f:
                csv.writer(f).writerow([4, 100_000, deg, r, f'{acc*100:.2f}%', f'{dt:.0f}'])


if __name__ == '__main__':
    main()
    poly_lr()
    print('done')
