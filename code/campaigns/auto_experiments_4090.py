"""RTX 4090 24GB 实验: 9-XOR 10M/12M + 10-XOR 10M/15M"""
import sys,os,csv,time,gc
import torch; import torch.nn as nn; import torch.nn.functional as F
import numpy as np
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from utils.PUFs import XORPUFModel
from utils.data import makeData  # CPU生成, 避免24GB OOM

class M(nn.Module):
    def __init__(self,k):
        super().__init__()
        d=[64,2**(k-1),2**k,2**(k-1),1]
        layers=[]
        for i in range(len(d)-1):
            layers.append(nn.Linear(d[i],d[i+1]))
            layers.append(nn.Tanh() if i<len(d)-2 else nn.Sigmoid())
        self.net=nn.Sequential(*layers)
    def forward(self,x): return self.net(x)

def run(xor,ds,rid,CSV):
    gc.collect(); torch.cuda.empty_cache()
    t0=time.time()
    d=makeData(XORPUFModel.randomSample(xor,64),ds); np.random.shuffle(d)
    N=len(d); tn=int(N*0.8)
    tp=torch.from_numpy(d[:tn,:-1]).float().cuda()
    tr=torch.from_numpy(d[:tn,-1:]).float().cuda()
    ttp=torch.from_numpy(d[tn:,:-1]).float()
    ttr=torch.from_numpy(d[tn:,-1:]).float()
    del d; gc.collect()
    m=M(xor).cuda()
    opt=torch.optim.Adam(m.parameters(),lr=0.001)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,500)
    best=0.0; st=time.time()
    for ep in range(1,501):
        m.train(); perm=torch.randperm(tn)
        for i in range(0,tn,16384):
            idx=perm[i:i+16384]; loss=F.binary_cross_entropy(m(tp[idx]),tr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
        # 每10个epoch在测试集上评估, 更新best
        if ep % 10 == 0:
            m.eval()
            with torch.no_grad():
                va=(m(ttp.cuda()).round()==ttr.cuda()).float().mean().item()
            best=max(best,va)
            if ep>150 and best<0.52: break
            m.train()
    dt=time.time()-st; m.eval()
    with torch.no_grad(): ta=(m(ttp.cuda()).round()==ttr.cuda()).float().mean().item()
    ok=1 if ta>0.9 else 0
    mark="✅" if ok else "❌"
    print(f'  #{rid}: {ta*100:.1f}% {dt:.0f}s {mark}')
    with open(CSV,'a') as f: csv.writer(f).writerow([xor,ds,rid,f'{ta*100:.2f}%',f'{dt:.0f}',ok])
    del m,tp,tr,ttp,ttr; gc.collect(); torch.cuda.empty_cache()
    return ok

# ==== Phase 1: 9-XOR 10M ×5 ====
CSV='results/9xor_10M.csv'
with open(CSV,'w') as f: csv.writer(f).writerow(['xor','data','run','acc','time','ok'])
print('=== Phase 1: 9-XOR 10M x5 ===')
for r in range(1,6):
    run(9,10_000_000,r,CSV)

# ==== Phase 2: 9-XOR 12M ×3 ====
CSV='results/9xor_12M.csv'
with open(CSV,'w') as f: csv.writer(f).writerow(['xor','data','run','acc','time','ok'])
print('=== Phase 2: 9-XOR 12M x3 ===')
for r in range(1,4):
    run(9,12_000_000,r,CSV)

# ==== Phase 3: 10-XOR 10M ×3 ====
CSV='results/10xor_10M.csv'
with open(CSV,'w') as f: csv.writer(f).writerow(['xor','data','run','acc','time','ok'])
print('=== Phase 3: 10-XOR 10M x3 ===')
for r in range(1,4):
    run(10,10_000_000,r,CSV)

# ==== Phase 4: 10-XOR 15M ×3 ====
CSV='results/10xor_15M.csv'
with open(CSV,'w') as f: csv.writer(f).writerow(['xor','data','run','acc','time','ok'])
print('=== Phase 4: 10-XOR 15M x3 ===')
for r in range(1,4):
    run(10,15_000_000,r,CSV)

print('All done')
