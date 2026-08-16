"""任务2补实验: 7-XOR 1.25M×5 + 6-XOR 200k×5 + 8-XOR 5M噪声ε=3%×5
标准协议: 4层MLP [64,2^(k-1),2^k,2^(k-1),1], Tanh+Sigmoid,
Adam lr=0.001, CosineAnnealing(500), batch=4096, 80/2/18划分,
验证集早停(每10epoch评估, ep>150且best<0.52则停)"""
import sys,os,csv,time,gc
import torch; import torch.nn as nn; import torch.nn.functional as F
import numpy as np
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from utils.PUFs import XORPUFModel

class M(nn.Module):
    def __init__(self,k):
        super().__init__()
        d=[64,2**(k-1),2**k,2**(k-1),1]; layers=[]
        for i in range(len(d)-1):
            layers.append(nn.Linear(d[i],d[i+1]))
            layers.append(nn.Tanh() if i<len(d)-2 else nn.Sigmoid())
        self.net=nn.Sequential(*layers)
    def forward(self,x): return self.net(x)

def batchResponse(puf, ch):
    w=puf.weight[:,:-1].astype(np.float32); b=puf.weight[:,-1].astype(np.float32)
    bits=(np.dot(ch,w.T)+b>=0).astype(np.int8)
    return np.bitwise_xor.reduce(bits,axis=1)

def run(xor, ds, rid, CSV, eps=None):
    gc.collect(); torch.cuda.empty_cache()
    puf=XORPUFModel.randomSample(xor,64)
    ch=np.random.randint(0,2,(ds,64)).astype(np.int8)*2-1
    clean=batchResponse(puf,ch).astype(np.float32)
    if eps is not None and eps>0:
        flip=np.random.rand(ds)<eps
        clean=clean.copy(); clean[flip]=1-clean[flip]
    d=np.column_stack([ch.astype(np.float32),clean.reshape(-1,1)])
    del ch,clean
    np.random.shuffle(d)
    tn=int(len(d)*0.8); vn=int(len(d)*0.02)
    tp=torch.from_numpy(d[:tn,:-1]).float().cuda(); tr=torch.from_numpy(d[:tn,-1:]).float().cuda()
    tv=torch.from_numpy(d[tn:tn+vn,:-1]).float().cuda(); tvr=torch.from_numpy(d[tn:tn+vn,-1:]).float().cuda()
    ttp=torch.from_numpy(d[tn+vn:,:-1]).float(); ttr=torch.from_numpy(d[tn+vn:,-1:]).float()
    del d; gc.collect()
    m=M(xor).cuda()
    opt=torch.optim.Adam(m.parameters(),lr=0.001)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,500)
    best=0.0; st=time.time()
    for ep in range(1,501):
        m.train(); perm=torch.randperm(tn)
        for i in range(0,tn,4096):
            idx=perm[i:i+4096]; loss=F.binary_cross_entropy(m(tp[idx]),tr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
        if ep%10==0:
            m.eval()
            with torch.no_grad(): va=(m(tv).round()==tvr).float().mean().item()
            best=max(best,va)
            if ep>150 and best<0.52: break
            m.train()
    dt=time.time()-st; m.eval()
    with torch.no_grad(): ta=(m(ttp.cuda()).round()==ttr.cuda()).float().mean().item()
    ok=1 if ta>0.9 else 0
    print(f'  {xor}X {ds//1000}k r{rid}: {ta*100:.1f}% {dt:.0f}s {"OK"if ok else "FAIL"}')
    row=[xor,ds,rid,f'{ta*100:.2f}%',f'{dt:.0f}',ok]
    if eps is not None:
        row=[xor,ds,f'{eps}',rid,f'{ta*100:.2f}%',f'{dt:.0f}',ok]
    with open(CSV,'a') as f: csv.writer(f).writerow(row)
    del m,tp,tr,tv,tvr,ttp,ttr; gc.collect(); torch.cuda.empty_cache()

# ==== 1. 7-XOR 1.25M runs 6-10 ====
CSV='/root/autodl-tmp/splitPUF/results/7xor_N50_fine.csv'
print('=== 7-XOR 1.25M runs 6-10 ===')
for r in range(6,11):
    run(7,1_250_000,r,CSV)

# ==== 2. 6-XOR 200k ×5 (新CSV) ====
CSV='/root/autodl-tmp/splitPUF/results/6xor_200k.csv'
with open(CSV,'w') as f: csv.writer(f).writerow(['xor','data','run','acc','time','ok'])
print('=== 6-XOR 200k x5 ===')
for r in range(1,6):
    run(6,200_000,r,CSV)

# ==== 3. 8-XOR 5M eps=3% runs 6-10 ====
CSV='/root/autodl-tmp/splitPUF/results/noise_8xor_5M.csv'
print('=== 8-XOR 5M eps=3% runs 6-10 ===')
for r in range(6,11):
    run(8,5_000_000,r,CSV,eps=0.03)

print('All done')
