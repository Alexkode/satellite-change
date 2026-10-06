import argparse,json,random,time
from pathlib import Path
import numpy as np,torch
from torch.utils.data import DataLoader,TensorDataset
from .data import load
from .models import ChangeNet

def confusion(y,p):
    y=np.asarray(y,dtype=bool);p=np.asarray(p,dtype=bool)
    return np.array([(y&p).sum(),((~y)&p).sum(),(y&(~p)).sum(),((~y)&(~p)).sum()],dtype=np.int64)
def metrics(counts):
    tp,fp,fn,tn=map(int,counts)
    return {'tp':tp,'fp':fp,'fn':fn,'tn':tn,'precision':tp/max(tp+fp,1),'recall':tp/max(tp+fn,1),'f1':2*tp/max(2*tp+fp+fn,1),'iou':tp/max(tp+fp+fn,1),'accuracy':(tp+tn)/max(tp+fp+fn+tn,1)}

def infer(model,x,tile=128,device='cpu'):
    _,h,w=x.shape;out=np.empty((h,w),np.float32)
    model.eval()
    with torch.no_grad():
        for i in range(0,h,tile):
            for j in range(0,w,tile):
                patch=x[:,i:i+tile,j:j+tile];ph,pw=patch.shape[-2:]
                padded=np.pad(patch,((0,0),(0,max(0,4-ph)),(0,max(0,4-pw))),mode='edge')
                out[i:i+ph,j:j+pw]=model(torch.from_numpy(padded[None]).to(device)).sigmoid()[0,0,:ph,:pw].cpu().numpy()
    return out

def evaluate(scores,samples,threshold):
    per_city={c:metrics(confusion(samples[c][1],p>=threshold)) for c,p in scores.items()}
    counts=sum((confusion(samples[c][1],p>=threshold) for c,p in scores.items()),np.zeros(4,np.int64))
    return {'micro':metrics(counts),'macro_city_f1':float(np.mean([v['f1'] for v in per_city.values()])),'per_city':per_city}

def select(scores,samples,thresholds):
    values=[(evaluate(scores,samples,t)['macro_city_f1'],float(t)) for t in thresholds]
    return max(values,key=lambda x:x[0])

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['difference','unet','siamese'],default='difference');p.add_argument('--seed',type=int,default=0)
    p.add_argument('--epochs',type=int,default=15);p.add_argument('--patches-per-city',type=int,default=12);p.add_argument('--device',default='cpu');a=p.parse_args()
    torch.set_num_threads(2);torch.manual_seed(a.seed);np.random.seed(a.seed);random.seed(a.seed)
    split=json.loads(Path('results/split-manifest.json').read_text());train=load(split['train']);val=load(split['validation']);test=load(split['test'])
    output=Path(f'results/{a.model}-seed{a.seed}');output.mkdir(parents=True,exist_ok=True);start=time.perf_counter();history=[]
    if a.model=='difference':
        score=lambda samples:{c:np.sqrt(((x[:4]-x[4:])**2).mean(0)) for c,(x,y) in samples.items()}
        vs=score(val);ts=score(test);best,threshold=select(vs,val,np.linspace(.005,.3,60))
    else:
        xs=[];ys=[];rng=np.random.default_rng(a.seed)
        for x,y in train.values():
            h,w=y.shape
            for _ in range(a.patches_per_city):
                i=rng.integers(0,h-64+1);j=rng.integers(0,w-64+1);xx=x[:,i:i+64,j:j+64].copy();yy=y[None,i:i+64,j:j+64].copy()
                if rng.random()<.5:xx=xx[:,:,::-1].copy();yy=yy[:,:,::-1].copy()
                xs.append(xx);ys.append(yy)
        x=torch.from_numpy(np.stack(xs));y=torch.from_numpy(np.stack(ys).astype(np.float32));fraction=float(y.mean())
        loader=DataLoader(TensorDataset(x,y),batch_size=8,shuffle=True,generator=torch.Generator().manual_seed(a.seed))
        model=ChangeNet(a.model=='siamese').to(a.device);optim=torch.optim.Adam(model.parameters(),lr=.001)
        loss_fn=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(min(20,(1-fraction)/max(fraction,1e-6)),device=a.device))
        best_loss=float('inf');best_state=None
        for epoch in range(a.epochs):
            model.train();losses=[]
            for xx,yy in loader:
                xx,yy=xx.to(a.device),yy.to(a.device);logits=model(xx);loss=loss_fn(logits,yy)
                optim.zero_grad();loss.backward();optim.step();losses.append(float(loss.detach()))
            vs={c:infer(model,xx,device=a.device) for c,(xx,yy) in val.items()}
            # Validation BCE selects epoch independently of the test set.
            vl=float(np.mean([-np.mean(yy*np.log(np.clip(vs[c],1e-6,1))+(1-yy)*np.log(np.clip(1-vs[c],1e-6,1))) for c,(xx,yy) in val.items()]))
            history.append({'epoch':epoch+1,'train_weighted_bce':float(np.mean(losses)),'validation_bce':vl})
            if vl<best_loss:best_loss=vl;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        model.load_state_dict(best_state);vs={c:infer(model,xx,device=a.device) for c,(xx,yy) in val.items()};best,threshold=select(vs,val,np.linspace(.1,.9,17))
        ts={c:infer(model,xx,device=a.device) for c,(xx,yy) in test.items()};torch.save({'state':best_state,'model':a.model},output/'checkpoint.pt')
    result={'config':vars(a),'torch':torch.__version__,'seconds':time.perf_counter()-start,'threshold':threshold,'validation_macro_f1':best,
            'test':evaluate(ts,test,threshold),'history':history,'method':'All official test cities held out. Saclay east excluded because west is test. Last 3 eligible training cities form validation. All 10m image pixels evaluated by nonoverlapping tiles; no random pixel split.'}
    (output/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    city=sorted(ts)[0];np.savez_compressed(output/'example.npz',score=ts[city],truth=test[city][1],before=test[city][0][[2,1,0]],after=test[city][0][[6,5,4]],city=city)
    print(json.dumps({'model':a.model,'seed':a.seed,'threshold':threshold,'test':result['test']['micro'],'macro':result['test']['macro_city_f1']}))

if __name__=='__main__':main()
