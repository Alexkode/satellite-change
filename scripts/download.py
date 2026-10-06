import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from change_lab.data import ARCHIVES
import httpx
root=Path('data');root.mkdir(exist_ok=True)
with httpx.Client(timeout=120,follow_redirects=True) as client:
    for name,(url,_) in ARCHIVES.items():
        path=root/name
        if path.exists():continue
        temp=path.with_suffix('.partial')
        with client.stream('GET',url) as response:
            response.raise_for_status()
            with temp.open('wb') as out:
                for chunk in response.iter_bytes():out.write(chunk)
        temp.replace(path)
        print('Downloaded',name)
