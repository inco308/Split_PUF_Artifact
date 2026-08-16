"""GPU优化续跑: 架构独立性+7-XOR补充"""
import sys,os,csv,time,gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
os.environ['CUDA_VISIBLE_DEVICES']='0'
from utils.PUFs import XORPUFModel
from utils.data_fast import makeData_fast as makeData

class M(nn.Module):
    def __init__(self,k):
        super().__init__()
        d=[64,2**(k-1),2**k,2**(k-1),1]
        layers=[]
        for i in range(len(d)-1):
            layers.append(nn.Linear(d[i],d[i+1]))
            if i<len(d)-2: layers.append(nn.Tanh())
            else: layers.append(nn.Sigmoid())
        self.net=nn.Sequential(*layers)
    def forward(self,x): return self.net(x)

def run_one(xor,ds,rid,CSV):
    gc.collect(); torch.cuda.empty_cache()
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
        if ep>150 and best<0.52: break
    dt=time.time()-st; m.eval()
    with torch.no_grad(): ta=(m(ttp.cuda()).round()==ttr.cuda()).float().mean().item()
    ok=1 if ta>0.9 else 0
    print(f'  r{rid}: {ta*100:.1f}% {dt:.0f}s {"✅"if ok else "❌"}')
    with open(CSV,'a') as f: csv.writer(f).writerow([xor,ds,rid,f'{ta*100:.2f}%',f'{dt:.0f}',ok])
    del m,tp,tr,ttp,ttr; gc.collect(); torch.cuda.empty_cache()
    return ok

# ==== 1. 架构无关性: 6-XOR 300k ×5 ====
CSV='results/arch_indep_6xor_below.csv'
with open(CSV,'w') as f: csv.writer(f).writerow(['xor','data','run','acc','time','ok'])
print('=== 架构无关性: 6-XOR 300k ×5 ===')
for r in range(1,6):
    run_one(6,300_000,r,CSV)

# ==== 2. 7-XOR: 500k/1M/2M/5M ×5 ====
CSV='results/7xor_baseline.csv'
with open(CSV,'w') as f: csv.writer(f).writerow(['xor','data','run','acc','time','ok'])
for ds in [500_000,1_000_000,2_000_000,5_000_000]:
    print(f'=== 7-XOR {ds//1000}k ×5 ===')
    for r in range(1,6):
        run_one(7,ds,r,CSV)

print('Done')
