"""8-XOR 10M复现: N=1→N=10"""
import sys,os,csv,time,gc,torch,torch.nn as nn,torch.nn.functional as F,numpy as np
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

CSV='results/8xor_10M_replication.csv'
with open(CSV,'w') as f: csv.writer(f).writerow(['xor','data','run','acc','time','ok'])

for r in range(1,10):
    gc.collect(); torch.cuda.empty_cache()
    puf=XORPUFModel.randomSample(8,64); d=makeData(puf,10_000_000); np.random.shuffle(d)
    N=len(d); tn=int(N*0.8)
    tp=torch.from_numpy(d[:tn,:-1]).float(); tr=torch.from_numpy(d[:tn,-1:]).float()
    ttp=torch.from_numpy(d[tn:,:-1]).float(); ttr=torch.from_numpy(d[tn:,-1:]).float()
    m=M(8).cuda(); tp=tp.cuda(); tr=tr.cuda()
    opt=torch.optim.Adam(m.parameters(),lr=0.001)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,500)
    best=0.0; st=time.time()
    for ep in range(1,501):
        m.train(); perm=torch.randperm(tn)
        for i in range(0,tn,4096):
            idx=perm[i:i+4096]; loss=F.binary_cross_entropy(m(tp[idx]),tr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
        if ep>150 and best<0.52: break
    dt=time.time()-st; m.eval()
    with torch.no_grad(): ta=(m(ttp.cuda()).round()==ttr.cuda()).float().mean().item()
    ok=1 if ta>0.9 else 0
    print(f'  r{r}: {ta*100:.1f}% {dt:.0f}s {"✅"if ok else "❌"}')
    with open(CSV,'a') as f: csv.writer(f).writerow([8,10000000,r,f'{ta*100:.2f}%',f'{dt:.0f}',ok])
print('Done')
