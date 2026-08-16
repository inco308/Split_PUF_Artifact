"""进化策略基线 v2: 软比特XOR对数似然目标 (光滑, 全局最优=真延迟)
PSO + CMA-ES 攻击 8-XOR APUF, 520维延迟参数化
P(r=1) = 0.5*(1 - Π_k(1-2σ(D_k))) — 独立软比特XOR的精确公式
目标: 最小化 NLL = -Σ [r·logP1 + (1-r)·logP0] / n"""
import sys,os,csv,time,gc
import numpy as np
import torch
import cma
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from utils.PUFs import XORPUFModel

K=8
CSV='/root/autodl-tmp/splitPUF/results/evolution_baselines.csv'
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
torch.set_grad_enabled(False)

def done_set():
    if not os.path.exists(CSV): return set()
    return {(r['xor'],r['data'],r['method'],r['run']) for r in csv.DictReader(open(CSV))}

def make_data(ds):
    puf=XORPUFModel.randomSample(K,64)
    ch=np.random.randint(0,2,(ds,64)).astype(np.float32)*2-1
    w=puf.weight[:,:-1].astype(np.float32); b=puf.weight[:,-1].astype(np.float32)
    resp=(np.bitwise_xor.reduce(((ch@w.T)+b>=0).astype(np.int8),axis=1)).astype(np.float32)
    perm=np.random.permutation(ds)
    ch,resp=ch[perm],resp[perm]
    return ch,resp

def nll_and_acc(W, C, Y):
    """W: (P,K,65) tensor on DEVICE. 返回 (NLL(P,), train_acc(P,))"""
    wd=W[:,:,:64]; b=W[:,:,64]
    D=torch.einsum('nd,pkd->pnk',C,wd)+b.unsqueeze(1)      # (p,n,k)
    s=torch.sigmoid(D)                                     # 软比特
    prod=(1-2*s).prod(dim=2)                               # (p,n)
    P1=0.5*(1-prod); P1=P1.clamp(1e-7,1-1e-7)
    P0=1-P1
    Yb=Y.unsqueeze(0)
    nll=-(Yb*P1.log()+(1-Yb)*P0.log()).mean(dim=1)         # (p,)
    parity=(D>=0).int().sum(dim=2)%2
    acc=(parity.float()==Yb).float().mean(dim=1)
    return nll,acc

def pso_attack(ch,resp):
    ds_key=str(ch.shape[0]); tn=int(len(ch)*0.8)
    for run in range(1,3):
        if ('8',ds_key,'PSO',str(run)) in done_set():
            print(f'  skip PSO {ds_key} run{run} (done)'); continue
        gc.collect(); torch.cuda.empty_cache(); torch.manual_seed(100+run)
        C=torch.from_numpy(ch).to(DEVICE); Y=torch.from_numpy(resp).to(DEVICE)
        C_tr,Y_tr,C_te,Y_te=C[:tn],Y[:tn],C[tn:],Y[tn:]
        pop=20; W=(torch.randn(pop,K,65,device=DEVICE)*0.1); V=torch.zeros_like(W)
        pbest=W.clone()
        pbest_nll,_=nll_and_acc(W,C_tr,Y_tr)
        gi=pbest_nll.argmin(); gbest_nll=pbest_nll[gi].item(); gbest=pbest[gi].clone()
        w,c1,c2=0.72,1.49,1.49; st=time.time()
        for it in range(5000):
            r1=torch.rand(pop,K,65,device=DEVICE); r2=torch.rand(pop,K,65,device=DEVICE)
            V=w*V+c1*r1*(pbest-W)+c2*r2*(gbest.unsqueeze(0)-W); W=W+V
            nll,_=nll_and_acc(W,C_tr,Y_tr)
            mask=nll<pbest_nll; pbest[mask]=W[mask]; pbest_nll[mask]=nll[mask]
            gi=nll.argmin()
            if nll[gi]<gbest_nll: gbest_nll=nll[gi].item(); gbest=W[gi].clone()
            if it%100==0:
                ga=nll_and_acc(gbest.unsqueeze(0),C_tr,Y_tr)[1][0].item()
                print(f'    PSO it{it}: NLL={gbest_nll:.4f} tr_acc={ga*100:.2f}% ({time.time()-st:.0f}s)')
        _,tr_acc=nll_and_acc(gbest.unsqueeze(0),C_tr,Y_tr)
        _,te_acc=nll_and_acc(gbest.unsqueeze(0),C_te,Y_te)
        print(f'  PSO {ds_key} run{run}: NLL={gbest_nll:.4f} train={tr_acc[0].item()*100:.2f}% test={te_acc[0].item()*100:.2f}%')
        with open(CSV,'a') as f: csv.writer(f).writerow([8,int(ds_key),'PSO',run,f'{gbest_nll:.4f}',f'{tr_acc[0].item()*100:.2f}%',f'{te_acc[0].item()*100:.2f}%',f'{time.time()-st:.0f}',''])
        del C,Y,C_tr,Y_tr,C_te,Y_te,W,V,pbest; gc.collect(); torch.cuda.empty_cache()

def cma_attack(ch,resp,budget_gen=10000):
    ds_key=str(ch.shape[0])
    if ('8',ds_key,'CMA-ES','1') in done_set():
        print(f'  skip CMA-ES {ds_key} (done)'); return
    tn=int(len(ch)*0.8)
    C_tr=torch.from_numpy(ch[:tn]).to(DEVICE); Y_tr=torch.from_numpy(resp[:tn]).to(DEVICE)
    C_te=torch.from_numpy(ch[tn:]).to(DEVICE); Y_te=torch.from_numpy(resp[tn:]).to(DEVICE)
    def obj(W):
        Wg=torch.from_numpy(np.asarray(W,dtype=np.float32).reshape(-1,K,65)).to(DEVICE)
        nll,_=nll_and_acc(Wg,C_tr,Y_tr)
        return nll.cpu().numpy()
    es=cma.CMAEvolutionStrategy(np.zeros(K*65),0.3,{'maxiter':budget_gen,'popsize':23,'verb_log':0})
    st=time.time(); best_nll=np.inf
    while not es.stop():
        X=es.ask(); fits=obj(X); es.tell(X,fits)
        cur=fits.min()
        if cur<best_nll:
            best_nll=cur
            if int(es.countiter)%100==0:
                Wb=torch.from_numpy(np.asarray(es.result.xbest,dtype=np.float32).reshape(1,K,65)).to(DEVICE)
                _,ga=nll_and_acc(Wb,C_tr,Y_tr)
                print(f'    CMA it{es.countiter}: NLL={best_nll:.4f} tr_acc={ga[0].item()*100:.2f}% ({time.time()-st:.0f}s)')
    Wbest=torch.from_numpy(np.asarray(es.result.xbest,dtype=np.float32).reshape(1,K,65)).to(DEVICE)
    _,tr_acc=nll_and_acc(Wbest,C_tr,Y_tr)
    _,te_acc=nll_and_acc(Wbest,C_te,Y_te)
    print(f'  CMA {ds_key}: NLL={best_nll:.4f} train={tr_acc[0].item()*100:.2f}% test={te_acc[0].item()*100:.2f}% iters={es.countiter}')
    with open(CSV,'a') as f: csv.writer(f).writerow([8,int(ds_key),'CMA-ES',1,f'{best_nll:.4f}',f'{tr_acc[0].item()*100:.2f}%',f'{te_acc[0].item()*100:.2f}%',f'{time.time()-st:.0f}',es.countiter])
    del C_tr,Y_tr,C_te,Y_te; gc.collect(); torch.cuda.empty_cache()

for ds in [2_000_000,5_000_000]:
    print(f'=== 8-XOR {ds//1e6}M ===')
    ch,resp=make_data(ds)
    print(f'  data ready ({ch.shape[0]} rows)')
    pso_attack(ch,resp)
    cma_attack(ch,resp)
    del ch,resp; gc.collect()
print('All done')
