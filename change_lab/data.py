import argparse,hashlib,io,json,zipfile
from pathlib import Path
import numpy as np
from PIL import Image

BANDS=['B02','B03','B04','B08']
ARCHIVES={'images.zip':('https://partage.imt.fr/index.php/s/gKRaWgRnLMfwMGo/download','c50d4a2941da64e03a47ac4dec63d915'),
          'train-labels.zip':('https://partage.imt.fr/index.php/s/2D6n03k58ygBSpu/download','4d2965af8170c705ebad3d6ee71b6990'),
          'test-labels.zip':('https://partage.imt.fr/index.php/s/gpStKn4Mpgfnr63/download','8177d437793c522653c442aa4e66c617')}

def read_image(archive,name):
    with archive.open(name) as f:
        with Image.open(io.BytesIO(f.read())) as im:return np.asarray(im).copy()

def prepare(root=Path('data')):
    manifest=[]
    for name,(url,expected) in ARCHIVES.items():
        p=root/name;raw=p.read_bytes()
        if hashlib.md5(raw).hexdigest()!=expected:raise ValueError('Archive does not match published TorchGeo MD5: '+name)
        manifest.append({'archive':name,'url':url,'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'md5_verified':expected})
    samples={};groups={};extents={}
    with zipfile.ZipFile(root/'images.zip') as images:
        names={n.replace('\\','/'):n for n in images.namelist()}
        for group in ['train','test']:
            cities=[]
            with zipfile.ZipFile(root/(group+'-labels.zip')) as labels:
                mask_names=[n for n in labels.namelist() if n.replace('\\','/').endswith('/cm/cm.png')]
                for mask_name in mask_names:
                    city=mask_name.replace('\\','/').split('/')[-3];cities.append(city);arrays=[];rectified=[]
                    for time in ['imgs_1','imgs_2']:
                        channels=[];reference=[]
                        for band in BANDS:
                            matches=[original for normalized,original in names.items() if '/'+city+'/'+time+'/' in normalized and normalized.endswith('_'+band+'.tif')]
                            if len(matches)!=1:raise ValueError(f'Expected one {city}/{time}/{band}, found {matches}')
                            channels.append(read_image(images,matches[0]).astype(np.float32))
                            rect=[original for normalized,original in names.items() if '/'+city+'/'+time+'_rect/'+band+'.tif' in normalized]
                            if len(rect)!=1:raise ValueError('Missing rectified reference band')
                            reference.append(read_image(images,rect[0]).astype(np.float32))
                        arrays.append(np.stack(channels))
                        rectified.append(np.stack(reference))
                    mask=read_image(labels,mask_name)
                    if mask.ndim==3:mask=mask[...,0]
                    h=min(mask.shape[0],arrays[0].shape[1],arrays[1].shape[1]);w=min(mask.shape[1],arrays[0].shape[2],arrays[1].shape[2])
                    for original,reference in zip(arrays,rectified):
                        if not np.array_equal(original[:,:h,:w],reference[:,:h,:w]):
                            raise ValueError('Native bands disagree with rectified inputs: '+city)
                    extents[city]={'label_shape':list(mask.shape[:2]),'evaluated_shape':[h,w],
                                   'excluded_border_pixels':int(mask.shape[0]*mask.shape[1]-h*w)}
                    x=np.concatenate([a[:,:h,:w] for a in arrays]);y=(mask[:h,:w]>0).astype(np.uint8)
                    # Four native 10m bands; physical reflectance scale, no test-fitted statistics.
                    samples[city]=(np.clip(x/10000,0,1).astype(np.float32),y)
                    np.savez_compressed(root/(city+'.npz'),image=samples[city][0],mask=y)
            groups[group]=sorted(cities)
    # Saclay east and west are neighboring parts of the same geographic area.
    # West is official test: exclude east from both fitting and model selection.
    eligible=[c for c in groups['train'] if c!='saclay_e']
    train=eligible[:-3];validation=eligible[-3:];test=groups['test']
    assert not(set(train)&set(validation) or set(train)&set(test) or set(validation)&set(test))
    assert not({c.split('_')[0] for c in train+validation}&{c.split('_')[0] for c in test})
    split={'train':train,'validation':validation,'test':test,'excluded':['saclay_e'],'bands':BANDS,'reflectance_divisor':10000,
           'rectified_comparison':{'cities_checked':len(samples),'different_fields':0,'extent':'common native image and label overlap'},
           'archives':manifest,'cities':{c:{'shape':list(y.shape),'change_fraction':float(y.mean()),**extents[c]} for c,(x,y) in samples.items()}}
    Path('results/split-manifest.json').write_text(json.dumps(split,indent=2)+'\n');print(json.dumps(split))
    return split

def load(split):
    result={}
    for c in split:
        a=np.load('data/'+c+'.npz');result[c]=(a['image'],a['mask'])
    return result

if __name__=='__main__':prepare()
