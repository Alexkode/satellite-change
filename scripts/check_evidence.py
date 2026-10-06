"""Validate the configurations in the published experimental records."""
import json
from pathlib import Path
from datetime import datetime

root = Path(__file__).resolve().parents[1]
def read(path):
    return json.loads(path.read_text())

sat=root/'results';split=read(sat/'split-manifest.json')
assert split['rectified_comparison']['different_fields']==0
assert not(set(split['train'])&set(split['validation']) or set(split['train'])&set(split['test']))
assert not({c.split('_')[0] for c in split['train']+split['validation']}&{c.split('_')[0] for c in split['test']})
for model in ['unet','siamese']:
    for seed in [0,1,2]:
        r=read(sat/f'{model}-seed{seed}/metrics.json');assert set(r['test']['per_city'])==set(split['test'])
        assert r['config']['seed']==seed
print('Published evidence checks passed for satellite-change.')
