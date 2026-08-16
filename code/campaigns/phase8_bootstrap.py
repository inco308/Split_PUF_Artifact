"""Phase8: Bootstrap采样 — 关键相变点各10次"""
import sys,os,csv,torch,torch.nn as nn,torch.nn.functional as F,numpy as np,time,gc
sys.path.insert(0,'/root/autodl-tmp/splitPUF'); os.environ['CUDA_VISIBLE_DEVICES']='0'
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

CSV='results/phase8_bootstrap.csv'
with open(CSV,'w') as f: csv.writer(f).writerow(['xor','data','run','test','ok'])

def run(xor,ds,rid):
    try:
        gc.collect(); torch.cuda.empty_cache()
        puf=XORPUFModel.randomSample(xor,64); d=makeData(puf,ds); np.random.shuffle(d)
        N=len(d); tn,vn=int(N*0.8),int(N*0.02)
        tp=torch.from_numpy(d[:tn,:-1]).float(); tr=torch.from_numpy(d[:tn,-1:]).float()
        vp=torch.from_numpy(d[tn:tn+vn,:-1]).float().cuda(); vr=torch.from_numpy(d[tn:tn+vn,-1:]).float().cuda()
        ttp=torch.from_numpy(d[tn+vn:,:-1]).float(); ttr=torch.from_numpy(d[tn+vn:,-1:]).float()
        m=M(xor).cuda(); opt=torch.optim.Adam(m.parameters(),lr=0.001)
        sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,500)
        best=0.0; st=time.time()
        for ep in range(1,501):
            m.train(); perm=torch.randperm(tn)
            for i in range(0,tn,4096):
                idx=perm[i:i+4096]; loss=F.binary_cross_entropy(m(tp[idx].cuda()),tr[idx].cuda())
                opt.zero_grad(); loss.backward(); opt.step()
            sch.step(); m.eval()
            with torch.no_grad(): acc=(m(vp).round()==vr).float().mean().item()
            if acc>best: best=acc
            if ep>150 and best<0.52: break
        dt=time.time()-st
        with torch.no_grad(): ta=(m(ttp.cuda()).round()==ttr.cuda()).float().mean().item()
        ok=1 if ta>0.9 else 0
        print(f'  {xor}X {ds//1000}k r{rid}: {ta*100:.2f}% {dt:.0f}s {"✅"if ok else "❌"}')
        with open(CSV,'a') as f: csv.writer(f).writerow([xor,ds,rid,f'{ta*100:.2f}%',ok])
    except Exception as e: print(f'  ❌ {e}')

# 关键点各10次
configs=[(6,300_000),(6,350_000),(6,400_000),(8,5_000_000),(8,5_200_000),(8,5_400_000)]
for xor,ds in configs:
    for r in range(1,11):
        run(xor,ds,r); gc.collect(); torch.cuda.empty_cache()
print('Done')
