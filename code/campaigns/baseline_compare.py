"""MLP vs LR 对比实验 — 经典基线"""
import sys,os,csv,time,gc,torch,torch.nn as nn,torch.nn.functional as F,numpy as np
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
os.environ['CUDA_VISIBLE_DEVICES']='0'
from utils.PUFs import XORPUFModel
from utils.data_fast import makeData_fast as makeData
from sklearn.linear_model import SGDClassifier

class StdMLP(nn.Module):
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

def attack_mlp(xor,tp,tr,ttp,ttr,epochs=500):
    device='cuda'; m=StdMLP(xor).to(device)
    tp=tp.to(device); tr=tr.to(device)
    N=tp.shape[0]; opt=torch.optim.Adam(m.parameters(),lr=0.001)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs)
    best=0.0; st=time.time()
    for ep in range(1,epochs+1):
        m.train(); perm=torch.randperm(N)
        for i in range(0,N,4096):
            idx=perm[i:i+4096]; loss=F.binary_cross_entropy(m(tp[idx]),tr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
        if ep%200==0:
            m.eval()
            with torch.no_grad(): acc=(m(tp[:1000]).round()==tr[:1000]).float().mean().item()
            if acc>best: best=acc
        if ep>150 and best<0.52: break
    dt=time.time()-st; m.eval()
    with torch.no_grad(): ta=(m(ttp.to(device)).round()==ttr.to(device)).float().mean().item()
    return ta,dt

def attack_lr(tp,tr,ttp,ttr):
    st=time.time()
    X=((tp.numpy()+1)/2).astype(np.float32); y=tr.numpy().ravel().astype(int)
    clf=SGDClassifier(loss='log_loss',max_iter=1000,tol=1e-3,random_state=42)
    clf.fit(X,y); dt=time.time()-st
    Xt=((ttp.numpy()+1)/2).astype(np.float32); yt=ttr.numpy().ravel().astype(int)
    return clf.score(Xt,yt),dt

CSV='results/baseline_mlp_lr.csv'
with open(CSV,'w') as f: csv.writer(f).writerow(['xor','data','run','mlp_acc','mlp_time','lr_acc','lr_time'])

configs=[(6,300_000,10),(6,400_000,10),(6,500_000,10),(8,5_000_000,10),(8,5_500_000,10)]
for xor,ds,nr in configs:
    print(f'\n=== {xor}X {ds//1000}k ===')
    for r in range(1,nr+1):
        gc.collect(); torch.cuda.empty_cache()
        puf=XORPUFModel.randomSample(xor,64); d=makeData(puf,ds); np.random.shuffle(d)
        N=len(d); tn=int(N*0.8)
        tp=torch.from_numpy(d[:tn,:-1]).float(); tr=torch.from_numpy(d[:tn,-1:]).float()
        ttp=torch.from_numpy(d[tn:,:-1]).float(); ttr=torch.from_numpy(d[tn:,-1:]).float()

        ma,mt=attack_mlp(xor,tp,tr,ttp,ttr)
        la,lt=attack_lr(tp,tr,ttp,ttr) if ds<=2_000_000 else (0,0)

        print(f'  r{r}: MLP={ma*100:.1f}%({mt:.0f}s) LR={la*100:.1f}%({lt:.0f}s)')
        with open(CSV,'a') as f:
            csv.writer(f).writerow([xor,ds,r,f'{ma*100:.2f}%',f'{mt:.0f}',f'{la*100:.2f}%',f'{lt:.0f}'])
        gc.collect(); torch.cuda.empty_cache()
print('Done')
