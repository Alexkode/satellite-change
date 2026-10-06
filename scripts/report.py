import json,statistics
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path('results');split=json.loads((root/'split-manifest.json').read_text());lines=['# Satellite change detection experiment','',
    f"Official OSCD: {len(split['train'])} training cities, {len(split['validation'])} validation cities, {len(split['test'])} held-out test cities. Four native 10m Sentinel-2 bands. Small CPU models, 15 epochs, width 8, 12 random 64×64 patches per training city per seed.",'',
    '| Model | Micro F1 | Micro IoU | Macro city F1 |','|---|---:|---:|---:|']
summary={'split':{k:split[k] for k in ['train','validation','test']},'models':{}}
for name in ['difference','unet','siamese']:
    runs=[json.loads(p.read_text()) for p in sorted(root.glob(name+'-seed*/metrics.json'))]
    assert len(runs)==(1 if name=='difference' else 3)
    stats={key:{'mean':float(np.mean([r['test']['micro'][key] for r in runs])),
                'std_across_seeds':float(np.std([r['test']['micro'][key] for r in runs]))} for key in ['f1','iou','precision','recall']}
    stats['macro_city_f1']=float(np.mean([r['test']['macro_city_f1'] for r in runs]));summary['models'][name]=stats
    lines.append(f"| {name} | {stats['f1']['mean']:.3f} ± {stats['f1']['std_across_seeds']:.3f} | {stats['iou']['mean']:.3f} | {stats['macro_city_f1']:.3f} |")
    if name=='siamese':
        example=np.load(root/(name+'-seed0')/'example.npz');r=runs[0]
        fig,axes=plt.subplots(1,4,figsize=(13,3.6));images=[np.moveaxis(example['before'],0,-1),np.moveaxis(example['after'],0,-1),example['truth'],example['score']>=r['threshold']]
        for ax,image,title in zip(axes,images,['Before','After','Ground truth','Siamese prediction']):
            ax.imshow(np.clip(image*3,0,1) if image.ndim==3 else image,cmap='gray');ax.set_title(title);ax.axis('off')
        fig.suptitle(str(example['city'])+' · OSCD / Daudt et al. 2018 · Modified Copernicus imagery; labels CC BY-NC-SA',fontsize=10);fig.tight_layout();fig.savefig(root/'example.png',dpi=160)
        per=r['test']['per_city'];worst=sorted(per,key=lambda c:per[c]['f1'])[:3];lines+=['',f"Lowest F1 cities for Siamese seed 0: {', '.join(c+' ('+format(per[c]['f1'],'.3f')+')' for c in worst)}."]
excluded=sum(split['cities'][c].get('excluded_border_pixels',0) for c in split['test'])
lines+=['',f'The threshold and epoch are chosen on validation cities only. Every pixel in the common native image/label extent of each test city is evaluated. Unsupported outer borders exclude {excluded:,} labeled pixels; per-city extents are retained in the manifest. Random crops never cross split boundaries. Model seed variability and per-city scores expose geographic generalization limits.','',
        'These are small independent baseline implementations, not reproductions of published benchmark training. Nonoverlapping inference tiles can create seams; four-band inputs omit nine available channels. Limited patches and epochs constrain the results.','',
        'Image-derived artifacts in this results directory are subject to the OSCD dataset terms, separately from the MIT code license. Attribution: Daudt, Le Saux, Boulch and Gousseau, Urban Change Detection for Multispectral Earth Observation Using Convolutional Neural Networks, IGARSS 2018.']
(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(root/'REPORT.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
