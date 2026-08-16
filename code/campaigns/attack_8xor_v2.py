"""
8-XOR 全面攻坚 v2 — 数据缩放 + 训练范式 + 架构改进

策略矩阵:
  A路: 数据缩放      — 5M, 10M CRP (最佳模型 S5)
  B路: 训练范式      — CMA-ES, 预训练+微调, 渐进课程
  C路: 架构改进      — ReLU替代Tanh, 残差连接, 不同初始化
  D路: 损失改进      — BCEWithLogitsLoss (去掉Sigmoid)

全部并行执行，失败不中断，自动记录所有结果。
"""

import sys, os, csv, json, gc
from time import time, strftime
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from utils.PUFs import XORPUFModel
from utils.data import makeData, splitDataLoader

RESULTS_FILE = "results/8xor_v2_results.csv"
BATCH = 4096
XOR = 8
LEN = 64


# ============================================================
# 模型变体
# ============================================================

class MLP_ReLU(nn.Module):
    """ReLU激活 + Kaiming初始化"""
    def __init__(self, model_k=8):
        super().__init__()
        dims = [64, 2**(model_k-1), 2**model_k, 2**(model_k-1), 1]
        layers = []
        for i in range(len(dims)-2):
            layers += [nn.Linear(dims[i], dims[i+1]), nn.ReLU()]
        layers += [nn.Linear(dims[-2], dims[-1]), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)
        # Kaiming初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x): return self.net(x)


class MLP_Residual(nn.Module):
    """带残差连接的MLP"""
    def __init__(self, model_k=8):
        super().__init__()
        h = 2**model_k
        self.fc1 = nn.Linear(64, h)
        self.fc2 = nn.Linear(h, h)
        self.fc3 = nn.Linear(h, h)
        self.fc4 = nn.Linear(h, 1)
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.act(self.fc1(x))
        identity = out
        out = self.act(self.fc2(out)) + identity  # 残差
        identity = out
        out = self.act(self.fc3(out)) + identity  # 残差
        return torch.sigmoid(self.fc4(out))


class MLP_NoSigmoid(nn.Module):
    """无Sigmoid + BCEWithLogitsLoss (更稳定梯度)"""
    def __init__(self, model_k=8):
        super().__init__()
        dims = [64, 2**(model_k-1), 2**model_k, 2**(model_k-1), 1]
        layers = []
        for i in range(len(dims)-1):
            layers += [nn.Linear(dims[i], dims[i+1])]
            if i < len(dims)-2:
                layers += [nn.Tanh()]
        self.net = nn.Sequential(*layers)

    def forward(self, x): return self.net(x)  # 输出logits


class MLP_Standard(nn.Module):
    """标准4层Tanh MLP (与split_attack.py中NNModel_baseline完全一致)"""
    def __init__(self, model_k=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(64, 2**(model_k-1)), nn.Tanh(),
            nn.Linear(2**(model_k-1), 2**model_k), nn.Tanh(),
            nn.Linear(2**model_k, 2**(model_k-1)), nn.Tanh(),
            nn.Linear(2**(model_k-1), 1), nn.Sigmoid(),
        )

    def forward(self, x): return self.net(x)


# ============================================================
# 训练引擎
# ============================================================

def train_standard(model, train_phi, train_res, valid_phi, valid_res,
                   epochs, lr=0.001, use_nosigmoid_loss=False):
    """标准梯度训练"""
    device = "cuda"
    model = model.to(device)
    train_phi = train_phi.to(device)
    train_res = train_res.to(device)
    valid_phi = valid_phi.to(device)
    valid_res = valid_res.to(device)
    N = train_phi.shape[0]

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val = 0.0
    st = time()

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(N, device=device)

        for i in range(0, N, BATCH):
            idx = perm[i:i + BATCH]
            phi_b, res_b = train_phi[idx], train_res[idx]

            pred = model(phi_b)
            if use_nosigmoid_loss:
                loss = F.binary_cross_entropy_with_logits(pred, res_b)
            else:
                loss = F.binary_cross_entropy(pred, res_b)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()

        # 验证
        model.eval()
        with torch.no_grad():
            raw = model(valid_phi)
            if use_nosigmoid_loss:
                val_pred = torch.sigmoid(raw)
            else:
                val_pred = raw
            val_acc = (val_pred.round() == valid_res).float().mean().item()

        if val_acc > best_val:
            best_val = val_acc

        if epoch % 200 == 0:
            print(f"    E{epoch}: val={val_acc*100:.2f}% best={best_val*100:.2f}%")

        if epoch > 150 and best_val < 0.52:
            break

    dt = time() - st
    return best_val, dt


def train_from_pretrained(model, pretrain_phi, pretrain_res,
                          train_phi, train_res, valid_phi, valid_res,
                          pretrain_epochs=300, finetune_epochs=500):
    """预训练+微调: 先在6-XOR上预训练,再在8-XOR上微调"""
    device = "cuda"
    model = model.to(device)

    # 阶段1: 预训练 (6-XOR)
    print(f"    预训练阶段 ({pretrain_epochs}ep)...")
    pretrain_phi = pretrain_phi.to(device)
    pretrain_res = pretrain_res.to(device)
    N_pt = pretrain_phi.shape[0]

    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    for ep in range(1, pretrain_epochs + 1):
        model.train()
        perm = torch.randperm(N_pt, device=device)
        for i in range(0, N_pt, BATCH):
            idx = perm[i:i + BATCH]
            loss = F.binary_cross_entropy(model(pretrain_phi[idx]), pretrain_res[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()

    pretrain_acc = test_quick(model, pretrain_phi[:10000], pretrain_res[:10000])
    print(f"    预训练准确率: {pretrain_acc*100:.2f}%")

    # 阶段2: 微调 (8-XOR)
    print(f"    微调阶段 ({finetune_epochs}ep)...")
    train_phi = train_phi.to(device)
    train_res = train_res.to(device)
    valid_phi = valid_phi.to(device)
    valid_res = valid_res.to(device)
    N_ft = train_phi.shape[0]

    opt = torch.optim.Adam(model.parameters(), lr=0.0001)  # 更小lr
    best_val = 0.0
    st = time()

    for ep in range(1, finetune_epochs + 1):
        model.train()
        perm = torch.randperm(N_ft, device=device)
        for i in range(0, N_ft, BATCH):
            idx = perm[i:i + BATCH]
            loss = F.binary_cross_entropy(model(train_phi[idx]), train_res[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_acc = (model(valid_phi).round() == valid_res).float().mean().item()
        if val_acc > best_val:
            best_val = val_acc
        if ep % 200 == 0:
            print(f"    E{ep}: val={val_acc*100:.2f}% best={best_val*100:.2f}%")

    dt = time() - st
    return best_val, dt


def train_cmaes(model_class, train_phi, train_res, valid_phi, valid_res,
                popsize=50, generations=100):
    """
    CMA-ES进化策略训练 — 绕过梯度问题。
    仅优化最后一层权重（CMA-ES在高维空间效率低）。
    """
    device = "cuda"
    print(f"    CMA-ES: pop={popsize}, gen={generations}")

    # 先确定模型总参数量
    temp_model = model_class().to(device)
    n_params = sum(p.numel() for p in temp_model.parameters())
    print(f"    总参数: {n_params:,} (CMA-ES仅优化最后一层)")

    # 固定前面的层, 仅优化最后一层的权重和bias
    model = model_class().to(device)

    # 提取最后一层参数
    last_layer = None
    for name, param in model.named_parameters():
        if 'weight' in name:
            last_layer = param

    if last_layer is None:
        print("    ❌ 找不到最后一层")
        return 0.5, 0

    # 最后一层维度: (1, 2^(k-1)) → 256维 + bias
    w_shape = last_layer.shape
    n_opt = w_shape[0] * w_shape[1] + 1  # +bias

    import cma
    es = cma.CMAEvolutionStrategy(
        n_opt * [0.0], 0.5,
        {'popsize': popsize, 'maxiter': generations, 'verbose': -1}
    )

    train_phi = train_phi.to(device)
    train_res = train_res.to(device)
    valid_phi = valid_phi.to(device)
    valid_res = valid_res.to(device)

    best_val = 0.0
    st = time()

    while not es.stop():
        solutions = es.ask()
        fitnesses = []

        for sol in solutions:
            # 将解向量reshape为权重和bias
            w_flat = sol[:-1]
            bias = sol[-1]
            with torch.no_grad():
                # 更新最后一层
                for name, param in model.named_parameters():
                    if name.endswith('.weight') and param.shape == w_shape:
                        param.copy_(torch.tensor(w_flat, device=device).reshape(w_shape))
                # 找对应的bias
                for name, param in model.named_parameters():
                    if name.endswith('.bias') and param.numel() == 1:
                        param.copy_(torch.tensor([bias], device=device))

            # 评估: 训练集loss
            model.eval()
            with torch.no_grad():
                pred = model(train_phi[:5000])
                loss = F.binary_cross_entropy(pred, train_res[:5000]).item()
            fitnesses.append(loss)

        es.tell(solutions, fitnesses)

        # 用当前最佳解评估验证集
        with torch.no_grad():
            val_pred = model(valid_phi)
            val_acc = (val_pred.round() == valid_res).float().mean().item()
        if val_acc > best_val:
            best_val = val_acc

        if es.result.iteration % 20 == 0:
            print(f"    CMA gen{es.result.iteration}: val={val_acc*100:.2f}%")

    dt = time() - st
    return best_val, dt


def test_quick(model, phi, res):
    """快速测试"""
    device = "cuda"
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        pred = model(phi.to(device))
        return (pred.round() == res.to(device)).float().mean().item()


def save_result(name, best_val, test_acc, dt, params, data_size, note=""):
    os.makedirs("results", exist_ok=True)
    file_exists = os.path.exists(RESULTS_FILE)
    with open(RESULTS_FILE, "a", newline="") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(["timestamp","strategy","data_size","best_val","test_acc",
                       "time_s","params","success","note"])
        w.writerow([strftime("%Y-%m-%d %H:%M:%S"), name, data_size,
                   f"{best_val*100:.2f}%", f"{test_acc*100:.2f}%" if test_acc else "N/A",
                   f"{dt:.1f}", params, 1 if test_acc and test_acc > 0.90 else 0, note])


def prepare_data(data_size, xor_num=8):
    """生成指定大小的PUF数据"""
    puf = XORPUFModel.randomSample(xor_num, LEN)
    dataset = makeData(puf, data_size)
    np.random.shuffle(dataset)
    N = len(dataset)
    tn, vn = int(N*0.8), int(N*0.02)
    return (
        torch.from_numpy(dataset[:tn, :-1]).float(),
        torch.from_numpy(dataset[:tn, -1:]).float(),
        torch.from_numpy(dataset[tn:tn+vn, :-1]).float(),
        torch.from_numpy(dataset[tn:tn+vn, -1:]).float(),
        torch.from_numpy(dataset[tn+vn:, :-1]).float(),
        torch.from_numpy(dataset[tn+vn:, -1:]).float(),
    )


# ============================================================
# 主流程
# ============================================================

def main():
    print(f"{'='*60}")
    print(f"  8-XOR 全面攻坚 v2")
    print(f"  A:数据缩放 | B:训练范式 | C:架构改进 | D:损失改进")
    print(f"{'='*60}")

    all_start = time()

    # ====== A路: 数据缩放 ======
    for ds, label in [(5_000_000, "A1_5M"), (10_000_000, "A2_10M")]:
        print(f"\n{'='*50}")
        print(f"  [{label}] 数据缩放: {ds//1e6:.0f}M CRP")
        print(f"{'='*50}")
        try:
            tp, tr, vp, vr, ttp, ttr = prepare_data(ds)
            model = MLP_Standard(8)
            params = sum(p.numel() for p in model.parameters())
            bv, dt = train_standard(model, tp, tr, vp, vr, 500)
            ta = test_quick(model, ttp, ttr)
            ok = "✅" if ta > 0.9 else "❌"
            print(f"  {ok} {label}: val={bv*100:.2f}% test={ta*100:.2f}% time={dt:.0f}s")
            save_result(label, bv, ta, dt, params, ds)
        except Exception as e:
            print(f"  ❌ {label}: {e}")
            save_result(label, 0, 0, 0, 0, ds, str(e)[:100])

    # ====== B路: 预训练+微调 ======
    print(f"\n{'='*50}")
    print(f"  [B1] 预训练(6-XOR) → 微调(8-XOR)")
    print(f"{'='*50}")
    try:
        # 生成6-XOR预训练数据
        ptp, ptr, _, _, _, _ = prepare_data(600_000, xor_num=6)
        tp, tr, vp, vr, ttp, ttr = prepare_data(2_000_000)
        model = MLP_Standard(8)
        params = sum(p.numel() for p in model.parameters())
        bv, dt = train_from_pretrained(model, ptp, ptr, tp, tr, vp, vr)
        ta = test_quick(model, ttp, ttr)
        ok = "✅" if ta > 0.9 else "❌"
        print(f"  {ok} B1: val={bv*100:.2f}% test={ta*100:.2f}% time={dt:.0f}s")
        save_result("B1_pretrain_finetune", bv, ta, dt, params, 2_000_000)
    except Exception as e:
        print(f"  ❌ B1: {e}")
        save_result("B1_pretrain_finetune", 0, 0, 0, 0, 2_000_000, str(e)[:100])

    # ====== C路: 架构改进 ======
    arch_tests = [
        ("C1_ReLU_Kaiming", MLP_ReLU, {}),
        ("C2_Residual", MLP_Residual, {}),
        ("C3_NoSigmoid", MLP_NoSigmoid, {"use_nosigmoid_loss": True}),
        ("C4_Standard_baseline", MLP_Standard, {}),
    ]

    tp, tr, vp, vr, ttp, ttr = prepare_data(2_000_000)

    for name, model_cls, kwargs in arch_tests:
        print(f"\n{'='*50}")
        print(f"  [{name}]")
        print(f"{'='*50}")
        try:
            torch.cuda.empty_cache()
            model = model_cls()
            params = sum(p.numel() for p in model.parameters())
            use_ns = kwargs.get("use_nosigmoid_loss", False)
            bv, dt = train_standard(model, tp, tr, vp, vr, 500, use_nosigmoid_loss=use_ns)
            if use_ns:
                ta = test_quick(model, ttp, ttr)
                # NoSigmoid模型需要手动加sigmoid
                model_eval = model
                orig_forward = model_eval.forward
                model_eval.forward = lambda x: torch.sigmoid(orig_forward(x))
                ta = test_quick(model_eval, ttp, ttr)
            else:
                ta = test_quick(model, ttp, ttr)
            ok = "✅" if ta > 0.9 else "❌"
            print(f"  {ok} {name}: val={bv*100:.2f}% test={ta*100:.2f}% time={dt:.0f}s params={params:,}")
            save_result(name, bv, ta, dt, params, 2_000_000)
        except Exception as e:
            print(f"  ❌ {name}: {e}")
            save_result(name, 0, 0, 0, 0, 2_000_000, str(e)[:100])

    # ====== D路: CMA-ES ======
    print(f"\n{'='*50}")
    print(f"  [D1] CMA-ES进化策略")
    print(f"{'='*50}")
    try:
        tp, tr, vp, vr, ttp, ttr = prepare_data(2_000_000)
        bv, dt = train_cmaes(MLP_Standard, tp, tr, vp, vr, popsize=30, generations=80)
        # 用最佳CMA模型评估
        ok = "✅" if bv > 0.9 else "❌"
        print(f"  {ok} D1_CMAES: val={bv*100:.2f}% time={dt:.0f}s")
        save_result("D1_CMAES", bv, bv, dt, 0, 2_000_000)
    except ImportError:
        print(f"  ⚠️ cma库未安装, 跳过CMA-ES")
        save_result("D1_CMAES", 0, 0, 0, 0, 2_000_000, "cma_not_installed")
    except Exception as e:
        print(f"  ❌ D1: {e}")
        save_result("D1_CMAES", 0, 0, 0, 0, 2_000_000, str(e)[:100])

    # 汇总
    total_t = time() - all_start
    print(f"\n{'='*60}")
    print(f"  全部完成! 总耗时 {total_t/3600:.1f}h")
    print(f"  结果: {RESULTS_FILE}")
    print(f"{'='*60}")

    # 飞书通知
    try:
        from utils.feishu_notifier import get_notifier
        n = get_notifier()
        n.send_text(
            f"🏁 8-XOR v2 攻坚完成\n总耗时: {total_t/3600:.1f}h\n结果: {RESULTS_FILE}",
            "milestone"
        )
    except:
        pass


if __name__ == "__main__":
    main()
