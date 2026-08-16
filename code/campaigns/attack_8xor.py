"""
8-XOR 攻坚实验 — 系统性尝试突破当前攻击边界

策略矩阵:
  S1: 增大 epochs (1000, 2000)
  S2: 加深网络 (6层 vs 4层)
  S3: 加宽网络 (model_k=XOR_num+1)
  S4: 迁移学习 (6-XOR预训练 → 8-XOR微调)
  S5: 余弦退火学习率

运行: PYTHONUNBUFFERED=1 python attack_8xor.py
"""

import sys, os, csv
from time import time, strftime
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from utils.PUFs import XORPUFModel
from utils.data import makeData, splitDataLoader

RESULTS_FILE = "results/8xor_attack_results.csv"
BATCH_SIZE = 4096
DATA_SIZE = 2_000_000  # 2M CRP
XOR_NUM = 8
PUF_LENGTH = 64


# ============================================================
# 模型变体
# ============================================================

class DeepMLP(nn.Module):
    """6层 MLP — 比原版深 50%"""
    def __init__(self, input_dim=64, model_k=8):
        super().__init__()
        h1, h2, h3, h4 = 2**(model_k-1), 2**model_k, 2**(model_k-1), 2**(model_k-2)
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1), nn.Tanh(),
            nn.Linear(h1, h2), nn.Tanh(),
            nn.Linear(h2, h3), nn.Tanh(),
            nn.Linear(h3, h4), nn.Tanh(),
            nn.Linear(h4, h2), nn.Tanh(),
            nn.Linear(h2, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


class WideMLP(nn.Module):
    """加宽 MLP — model_k+1 的隐藏层宽度"""
    def __init__(self, input_dim=64, model_k=9):  # 8-XOR用k=9
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 2**(model_k-1)), nn.Tanh(),
            nn.Linear(2**(model_k-1), 2**model_k), nn.Tanh(),
            nn.Linear(2**model_k, 2**(model_k-1)), nn.Tanh(),
            nn.Linear(2**(model_k-1), 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# 训练引擎
# ============================================================

def train_model(model, train_phi, train_res, valid_phi, valid_res,
                epochs, lr=0.001, use_cosine=True, log_interval=None):
    """标准训练 + 余弦退火"""
    device = "cuda"
    model = model.to(device)
    train_phi = train_phi.to(device)
    train_res = train_res.to(device)
    valid_phi = valid_phi.to(device)
    valid_res = valid_res.to(device)
    N = train_phi.shape[0]

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if use_cosine:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    else:
        scheduler = None

    best_acc = 0.0
    st = time()
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(N, device=device)

        for i in range(0, N, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            phi_b, res_b = train_phi[idx], train_res[idx]

            pred = model(phi_b)
            loss = F.binary_cross_entropy(pred, res_b)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if scheduler:
            scheduler.step()

        # 验证
        model.eval()
        with torch.no_grad():
            pred = model(valid_phi)
            val_acc = (pred.round() == valid_res).float().mean().item()

        history.append(val_acc)
        if val_acc > best_acc:
            best_acc = val_acc

        if log_interval and epoch % log_interval == 0:
            lr_now = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch:4d} | Val: {val_acc*100:6.2f}% | "
                  f"Best: {best_acc*100:6.2f}% | LR: {lr_now:.6f}")

        # 早停: 200 epochs 不涨
        if epoch > 200 and best_acc < 0.55:
            print(f"  ⚠️ 早停 @epoch {epoch}: best={best_acc*100:.2f}%")
            break

    dt = time() - st
    return best_acc, dt, history


def test_model(model, test_phi, test_res):
    """测试集评估"""
    device = "cuda"
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        pred = model(test_phi.to(device))
        acc = (pred.round() == test_res.to(device)).float().mean().item()
    return acc


def prepare_data():
    """生成8-XOR数据"""
    puf = XORPUFModel.randomSample(XOR_NUM, PUF_LENGTH)
    dataset = makeData(puf, DATA_SIZE)
    np.random.shuffle(dataset)

    N = len(dataset)
    train_n, valid_n = int(N * 0.8), int(N * 0.02)
    train_data = torch.from_numpy(dataset[:train_n]).float()
    valid_data = torch.from_numpy(dataset[train_n:train_n + valid_n]).float()
    test_data = torch.from_numpy(dataset[train_n + valid_n:]).float()

    return (train_data[:, :-1], train_data[:, -1:],
            valid_data[:, :-1], valid_data[:, -1:],
            test_data[:, :-1], test_data[:, -1:])


def save_result(strategy, epochs_used, best_val, test_acc, dt, params):
    """记录结果"""
    os.makedirs("results", exist_ok=True)
    file_exists = os.path.exists(RESULTS_FILE)
    with open(RESULTS_FILE, "a", newline="") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(["timestamp", "strategy", "epochs", "best_val", "test_acc",
                       "time_s", "params", "success"])
        w.writerow([strftime("%Y-%m-%d %H:%M:%S"), strategy, epochs_used,
                   f"{best_val*100:.2f}%", f"{test_acc*100:.2f}%",
                   f"{dt:.1f}", params, 1 if test_acc > 0.90 else 0])


# ============================================================
# 主流程
# ============================================================

def main():
    print(f"{'='*60}")
    print(f"  8-XOR 攻坚实验")
    print(f"  数据: {DATA_SIZE//1e6:.0f}M CRP, Epochs: 测试多种配置")
    print(f"{'='*60}")

    # 预生成数据 (所有策略共用)
    train_phi, train_res, valid_phi, valid_res, test_phi, test_res = prepare_data()
    print(f"  数据就绪: Train={train_phi.shape[0]}, Val={valid_phi.shape[0]}, Test={test_phi.shape[0]}")

    strategies = [
        # (名称, 模型构建函数, epochs, lr, cosine)
        ("S0_baseline_500ep", lambda: DeepMLP(64, 8), 500, 0.001, True),
        ("S1_more_epochs_1000", lambda: DeepMLP(64, 8), 1000, 0.001, True),
        ("S2_more_epochs_2000", lambda: DeepMLP(64, 8), 2000, 0.001, True),
        ("S3_deeper_6layer", lambda: DeepMLP(64, 8), 2000, 0.001, True),
        ("S4_wider_k9", lambda: WideMLP(64, 9), 2000, 0.001, True),
        ("S5_deep_wide_k9", lambda: DeepMLP(64, 9), 2000, 0.001, True),
        ("S6_lower_lr", lambda: DeepMLP(64, 8), 2000, 0.0005, True),
    ]

    for name, build_fn, epochs, lr, cosine in strategies:
        print(f"\n{'='*50}")
        print(f"  [{name}] epochs={epochs}, lr={lr}")
        print(f"{'='*50}")

        torch.cuda.empty_cache()
        model = build_fn()
        params = sum(p.numel() for p in model.parameters())
        print(f"  Params: {params:,}")

        try:
            best_val, dt, hist = train_model(
                model, train_phi, train_res, valid_phi, valid_res,
                epochs=epochs, lr=lr, use_cosine=cosine, log_interval=max(epochs//10, 100)
            )
            test_acc = test_model(model, test_phi, test_res)
            success = "✅" if test_acc > 0.90 else "❌"
            print(f"  {success} {name}: Val={best_val*100:.2f}%, Test={test_acc*100:.2f}%, Time={dt:.0f}s")
            save_result(name, epochs, best_val, test_acc, dt, params)
        except torch.cuda.OutOfMemoryError:
            print(f"  💥 OOM!")
            save_result(name, epochs, 0, 0, 0, params)

    # 汇总
    print(f"\n{'='*60}")
    print(f"  结果保存在 {RESULTS_FILE}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
