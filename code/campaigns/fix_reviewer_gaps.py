"""补充审稿人要求的缺失实验:
1. 8-XOR 6M复现×20 (修复非单调性)
2. 4-XOR 75k/90k ×10 (定位N50)
"""
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

def run(xor,ds,rid,CSV):
    gc.collect(); torch.cuda.empty_cache()
    puf=XORPUFModel.randomSample(xor,64); d=makeData(puf,ds); np.random.shuffle(d)
    N=len(d); tn=int(N*0.8)
    tp=torch.from_numpy(d[:tn,:-1]).float(); tr=torch.from_numpy(d[:tn,-1:]).float()
    ttp=torch.from_numpy(d[tn:,:-1]).float(); ttr=torch.from_numpy(d[tn:,-1:]).float()
    m=M(xor).cuda(); tp=tp.cuda(); tr=tr.cuda()
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
    print(f'  r{rid}: {ta*100:.1f}% {dt:.0f}s {"✅"if ok else "❌"}')
    with open(CSV,'a') as f: csv.writer(f).writerow([xor,ds,rid,f'{ta*100:.2f}%',f'{dt:.0f}',ok])

# 1. 8-XOR 6M ×20
CSV1='results/8xor_6M_replication.csv'
with open(CSV1,'w') as f: csv.writer(f).writerow(['xor','data','run','acc','time','ok'])
print('8-XOR 6M ×20')
for r in range(1,21):
    run(8,6_000_000,r,CSV1)

# 2. 4-XOR 75k + 90k ×10
CSV2='results/4xor_N50.csv'
with open(CSV2,'w') as f: csv.writer(f).writerow(['xor','data','run','acc','time','ok'])
for ds in [75000,90000]:
    print(f'4-XOR {ds//1000}k ×10')
    for r in range(1,11):
        run(4,ds,r,CSV2)

print('Done')
