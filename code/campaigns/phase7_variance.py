"""Phase7: 全面量化相变方差 — 6-XOR + 8-XOR 各4次重复"""
import sys,os,csv,torch,torch.nn as nn,torch.nn.functional as F,numpy as np,time,gc
sys.path.insert(0,'/root/autodl-tmp/splitPUF')
os.environ['CUDA_VISIBLE_DEVICES']='0'
from utils.PUFs import XORPUFModel
from utils.data_fast import makeData_fast as makeData

class M(nn.Module):
    def __init__(self,k):
        super().__init__()
        d=[64,2**(k-1),2**k,2**(k-1),1]
        self.net=nn.Sequential(*[x for i in range(len(d)-1) for x in [nn.Linear(d[i],d[i+1])]+([nn.Tanh()] if i<len(d)-2 else [nn.Sigmoid()])])
    def forward(self,x): return self.net(x)

CSV='results/phase7_variance.csv'
with open(CSV,'w') as f: csv.writer(f).writerow(['xor','data','run','val','test','time','ok'])

def run(xor,ds,run_id):
    name=f'{xor}XOR_{ds//1000}k_r{run_id}'
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
        print(f'  {name}: test={ta*100:.2f}% {dt:.0f}s {"✅"if ok else "❌"}')
        with open(CSV,'a') as f: csv.writer(f).writerow([xor,ds,run_id,f'{best*100:.1f}%',f'{ta*100:.2f}%',f'{dt:.0f}',ok])
        return ok
    except Exception as e: print(f'  ❌ {name}: {e}'); return 0

# 6-XOR: 300k/350k/400k/450k × 4 runs
print("=== 6-XOR Variance ===")
for ds in [300_000,350_000,400_000,450_000]:
    for r in range(1,5):
        run(6,ds,r); gc.collect(); torch.cuda.empty_cache()

# 8-XOR: 5.0M/5.2M/5.4M/5.5M × 4 runs
print("=== 8-XOR Variance ===")
for ds in [5_000_000,5_200_000,5_400_000,5_500_000]:
    for r in range(1,5):
        run(8,ds,r); gc.collect(); torch.cuda.empty_cache()

print('Done')
