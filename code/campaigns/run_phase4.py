"""
Phase 4: 精确定位 + iPUF + 大模型

基于Phase3结果自动选择下一批实验。
如果Phase3未完成则等待其CSV出现后自动推断。
"""

import sys, os, csv
from time import time, strftime
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from utils.PUFs import XORPUFModel
from utils.iPUF import IPUFModel
from utils.data_fast import makeData_fast as makeData

RESULTS = "results/phase4_results.csv"
BATCH = 4096
EPOCHS = 500


class StdMLP(nn.Module):
    def __init__(self, xor_num):
        super().__init__()
        k = xor_num
        self.net = nn.Sequential(
            nn.Linear(64, 2**(k-1)), nn.Tanh(),
            nn.Linear(2**(k-1), 2**k), nn.Tanh(),
            nn.Linear(2**k, 2**(k-1)), nn.Tanh(),
            nn.Linear(2**(k-1), 1), nn.Sigmoid(),
        )
    def forward(self, x): return self.net(x)


def train_eval(model, tp, tr, vp, vr, ttp, ttr):
    device = "cuda"
    model = model.to(device)
    tp= tp.to(device); tr= tr.to(device)
    vp= vp.to(device); vr= vr.to(device)
    N = tp.shape[0]

    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    best_val = 0.0; st = time()

    for ep in range(1, EPOCHS+1):
        model.train()
        perm = torch.randperm(N, device=device)
        for i in range(0, N, BATCH):
            idx = perm[i:i+BATCH]
            loss = F.binary_cross_entropy(model(tp[idx]), tr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
        model.eval()
        with torch.no_grad():
            acc = (model(vp).round() == vr).float().mean().item()
        if acc > best_val: best_val = acc
        if ep > 150 and best_val < 0.52: break

    dt = time() - st
    model.eval()
    with torch.no_grad():
        ta = (model(ttp.to(device)).round() == ttr.to(device)).float().mean().item()
    return best_val, ta, dt


def gen(xor_num, size):
    puf = XORPUFModel.randomSample(xor_num, 64)
    ds = makeData(puf, size); np.random.shuffle(ds)
    N = len(ds); tn, vn = int(N*0.8), int(N*0.02)
    return (torch.from_numpy(ds[:tn,:-1]).float(), torch.from_numpy(ds[:tn,-1:]).float(),
            torch.from_numpy(ds[tn:tn+vn,:-1]).float(), torch.from_numpy(ds[tn:tn+vn,-1:]).float(),
            torch.from_numpy(ds[tn+vn:,:-1]).float(), torch.from_numpy(ds[tn+vn:,-1:]).float())


def gen_ipuf(k_up, k_down, size):
    """生成iPUF数据"""
    ipuf = IPUFModel.randomSample(k_up=k_up, k_down=k_down, length=64)
    phi = np.random.choice([-1, 1], size=(size, 64))
    res = np.array([ipuf.getResponse(p) for p in phi])
    ds = np.column_stack([phi, res])
    np.random.shuffle(ds)
    N = len(ds); tn, vn = int(N*0.8), int(N*0.02)
    return (torch.from_numpy(ds[:tn,:-1]).float(), torch.from_numpy(ds[:tn,-1:]).float(),
            torch.from_numpy(ds[tn:tn+vn,:-1]).float(), torch.from_numpy(ds[tn:tn+vn,-1:]).float(),
            torch.from_numpy(ds[tn+vn:,:-1]).float(), torch.from_numpy(ds[tn+vn:,-1:]).float())


def save(name, xor, ds, bv, ta, dt, params, note=""):
    os.makedirs("results", exist_ok=True)
    fe = os.path.exists(RESULTS)
    with open(RESULTS, "a", newline="") as f:
        w = csv.writer(f)
        if not fe: w.writerow(["ts","name","xor","data","val","test","time","params","ok","note"])
        w.writerow([strftime("%H:%M:%S"), name, xor, ds, f"{bv*100:.1f}%", f"{ta*100:.2f}%",
                   f"{dt:.0f}", params, 1 if ta>0.9 else 0, note])


def main():
    print("Phase 4: 精确定位 + iPUF + 大模型\n")

    # ===== 1. 读取Phase3结果, 确定相变点附近需要加密测试 =====
    phase3_csv = Path("results/phase3_results.csv")
    phase_transition_low = 5  # 已知5M失败
    phase_transition_high = 10  # 已知10M成功

    if phase3_csv.exists():
        with open(phase3_csv) as f:
            for row in csv.DictReader(f):
                name = row.get('name','')
                test_acc = float(row.get('test','0%').replace('%',''))
                if '8XOR' in name:
                    ds = int(row['data'])
                    if test_acc > 90 and ds < phase_transition_high:
                        phase_transition_high = ds
                    if test_acc < 55 and ds > phase_transition_low:
                        phase_transition_low = ds

        print(f"Phase3推断: 相变点介于 {phase_transition_low//1e6:.0f}M-{phase_transition_high//1e6:.0f}M")

    # ===== 2. 加密测试相变点 =====
    low_m = phase_transition_low // 1_000_000
    high_m = phase_transition_high // 1_000_000
    if high_m - low_m > 1:
        step = max(1, (high_m - low_m) // 4)
        for ds_m in range(low_m + step, high_m, step):
            ds = ds_m * 1_000_000
            name = f"8XOR_fine_{ds_m}M"
            print(f"[{name}]")
            try:
                tp,tr,vp,vr,ttp,ttr = gen(8, ds)
                m = StdMLP(8)
                bv, ta, dt = train_eval(m, tp, tr, vp, vr, ttp, ttr)
                p = sum(p.numel() for p in m.parameters())
                ok = "✅" if ta>0.9 else "❌"
                print(f"  {ok} {ta*100:.2f}% {dt:.0f}s")
                save(name, 8, ds, bv, ta, dt, p)
            except Exception as e:
                print(f"  ❌ {e}"); save(name, 8, ds, 0, 0, 0, 0, str(e)[:50])
            torch.cuda.empty_cache()

    # ===== 3. iPUF实验 (终于!) =====
    ipuf_configs = [(2,2), (2,4), (4,2), (4,4)]
    for k_up, k_down in ipuf_configs:
        for ds_m in [1, 2, 5]:
            ds = ds_m * 1_000_000
            name = f"iPUF_{k_up}_{k_down}_{ds_m}M"
            print(f"[{name}]")
            try:
                tp,tr,vp,vr,ttp,ttr = gen_ipuf(k_up, k_down, ds)
                # iPUF等效XOR ≈ max(k_up, k_down) + min(k_up,k_down)/2
                # 用稍大的模型
                equiv_xor = max(k_up, k_down) + min(k_up, k_down)//2 + 1
                m = StdMLP(equiv_xor)
                bv, ta, dt = train_eval(m, tp, tr, vp, vr, ttp, ttr)
                p = sum(p.numel() for p in m.parameters())
                ok = "✅" if ta>0.9 else "❌"
                note = f"equiv_xor={equiv_xor}"
                print(f"  {ok} test={ta*100:.2f}% {dt:.0f}s params={p:,} {note}")
                save(name, equiv_xor, ds, bv, ta, dt, p, note)
            except Exception as e:
                print(f"  ❌ {e}"); save(name, 0, ds, 0, 0, 0, 0, str(e)[:50])
            torch.cuda.empty_cache()

    # ===== 4. 8-XOR大模型: 加宽后能否减少数据需求? =====
    print("[BigModel_8XOR_2M]")
    try:
        tp,tr,vp,vr,ttp,ttr = gen(8, 2_000_000)
        m = StdMLP(10)  # model_k=10 → dims 512,1024,512  ~1.2M params
        bv, ta, dt = train_eval(m, tp, tr, vp, vr, ttp, ttr)
        p = sum(p.numel() for p in m.parameters())
        ok = "✅" if ta>0.9 else "❌"
        print(f"  {ok} test={ta*100:.2f}% {dt:.0f}s params={p:,}")
        save("BigMLP_8XOR_2M", 8, 2_000_000, bv, ta, dt, p, "k=10_1.2Mparams")
    except torch.cuda.OutOfMemoryError:
        print(f"  💥 OOM"); save("BigMLP_8XOR_2M", 8, 2_000_000, 0,0,0,0,"OOM")
    except Exception as e:
        print(f"  ❌ {e}"); save("BigMLP_8XOR_2M", 8, 2_000_000, 0,0,0,0, str(e)[:50])

    print(f"\nDone. Results: {RESULTS}")


if __name__ == "__main__":
    main()
