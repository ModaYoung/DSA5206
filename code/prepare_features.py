"""Download the fixed 2020 cohort and extract frozen DenseNet features."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote
import hashlib, json, os, urllib.request, time
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
CACHE = ROOT / 'cache'
SHA = '196143d3dd32a2a43ad51d6891d460aaf690d741'
BASE = f'https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/{SHA}'

def download(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size:
        return
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={'User-Agent':'DSA5206-academic-reproduction'})
            with urllib.request.urlopen(req, timeout=90) as response:
                content = response.read()
            path.write_bytes(content)
            return
        except Exception:
            if attempt == 3:
                raise
            time.sleep(attempt + 1)

def main():
    DATA.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    download(BASE+'/metadata.csv', DATA/'metadata_snapshot.csv')
    # This label file is packaged with its original attribution and checksum.
    if not (DATA/'severity_scores.csv').exists():
        raise FileNotFoundError('Use the packaged data/severity_scores.csv.')
    meta = pd.read_csv(DATA/'metadata_snapshot.csv')
    labels = pd.read_csv(DATA/'severity_scores.csv', skiprows=5)
    cohort = meta.merge(labels, on='filename', how='inner', validate='one_to_one')
    assert len(cohort) == 94 and cohort.patientid.nunique() == 65
    assert cohort['view'].eq('PA').all()
    assert cohort[['geographic_mean','opacity_mean','patientid']].notna().all().all()
    assert cohort.filename.is_unique
    cohort.to_csv(DATA/'scored_metadata.csv', index=False)
    files = cohort.filename.tolist()
    def get_image(filename):
        path = CACHE/'images'/filename
        download(BASE+'/images/'+quote(filename, safe=''), path)
        return {'filename':filename,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size}
    with ThreadPoolExecutor(max_workers=6) as pool:
        manifest = list(pool.map(get_image, files))
    pd.DataFrame(manifest).to_csv(DATA/'image_manifest.csv', index=False)
    print('Downloaded and verified paths for 94 images', flush=True)
    import torch
    import torch.nn.functional as F
    import torchvision.transforms as transforms
    import torchxrayvision as xrv
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    torch.manual_seed(20261027)
    ds = xrv.datasets.COVID19_Dataset(
        imgpath=str(CACHE/'images'), csvpath=str(DATA/'scored_metadata.csv'),
        views=['PA'], transform=transforms.Compose([xrv.datasets.XRayCenterCrop(),xrv.datasets.XRayResizer(224)]))
    assert ds.csv.filename.tolist() == files
    model = xrv.models.DenseNet(weights='densenet121-res224-all',cache_dir=str(CACHE/'weights'),apply_sigmoid=False)
    model.op_threshs = None
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    names = list(model.pathologies)
    dl = torch.utils.data.DataLoader(ds,batch_size=8,shuffle=False,num_workers=0)
    logits,features = [],[]
    with torch.inference_mode():
        for i,batch in enumerate(dl):
            x = batch['img']
            h = F.adaptive_avg_pool2d(F.relu(model.features(x),inplace=False),(1,1)).flatten(1)
            z = model.classifier(h)
            logits.append(z.numpy())
            features.append(h.numpy())
            print(f'Extracted batch {i+1}/{len(dl)}',flush=True)
    z,h = np.concatenate(logits),np.concatenate(features)
    assert z.shape == (94,18) and h.shape == (94,1024)
    assert np.isfinite(z).all() and np.isfinite(h).all()
    np.savez_compressed(DATA/'frozen_features.npz',logits=z,features=h,pathologies=np.array(names))
    weight_files = list((CACHE/'weights').glob('*.pt'))
    provenance = {
        'dataset_repository':'https://github.com/ieee8023/covid-chestxray-dataset',
        'dataset_commit':SHA,'metadata_commit_date':'2020-03-30T23:28:06Z',
        'metadata_sha256':hashlib.sha256((DATA/'metadata_snapshot.csv').read_bytes()).hexdigest(),
        'scores_sha256':hashlib.sha256((DATA/'severity_scores.csv').read_bytes()).hexdigest(),
        'model':'densenet121-res224-all','feature_extractor':'frozen; eval mode; pre-sigmoid logits',
        'weights':[{ 'name':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in weight_files],
        'n_images':94,'n_patients':65,'pathologies':names,
        'feature_npz_sha256':hashlib.sha256((DATA/'frozen_features.npz').read_bytes()).hexdigest(),
        'not_identical_claim':'Historical cohort reconstructed from a fixed public snapshot; the team metadata_2020.csv was not available in its repository.'}
    (DATA/'provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    print(json.dumps(provenance,indent=2),flush=True)

if __name__ == '__main__':
    main()
