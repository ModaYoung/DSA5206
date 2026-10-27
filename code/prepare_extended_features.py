"""Extract frozen public BioViL, BioViL-T and RAD-DINO features.

Each model keeps its published inference preprocessing. No COVID severity
labels are used here. Hub revisions and file hashes are recorded for reruns.
"""
from pathlib import Path
import argparse, gc, hashlib, json, os, time, urllib.request
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'data'
CACHE = ROOT/'cache'/'weights'/'extended'
torch.set_num_threads(4)
torch.manual_seed(20261027)


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def resolve_revision(repo, ref='main'):
    url=f'https://huggingface.co/api/models/{repo}/revision/{ref}'
    with urllib.request.urlopen(url, timeout=60) as stream:
        return json.load(stream)['sha']


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', default=['biovil','biovil_t','rad_dino'])
    args=parser.parse_args()
    CACHE.mkdir(parents=True,exist_ok=True)
    meta=pd.read_csv(OUT/'scored_metadata.csv')
    files=meta.filename.tolist()
    provenance_path=OUT/'extended_feature_provenance.json'
    provenance=json.loads(provenance_path.read_text()) if provenance_path.exists() else {}
    from huggingface_hub import hf_hub_download
    for name in args.models:
        target=OUT/f'{name}_features.npz'
        if target.exists() and name in provenance:
            print(f'{name}: existing verified feature file {sha(target)}',flush=True)
            assert sha(target)==provenance[name]['features_sha256']
            continue
        started=time.monotonic()
        if name in ['biovil','biovil_t']:
            from health_multimodal.image.model.model import ImageModel
            from health_multimodal.image.model.types import ImageEncoderType
            from health_multimodal.image.data.io import load_image
            from health_multimodal.image.data.transforms import create_chest_xray_transform_for_inference
            temporal=name=='biovil_t'
            repo='microsoft/BiomedVLP-BioViL-T' if temporal else 'microsoft/BiomedVLP-CXR-BERT-specialized'
            ref='v1.0' if temporal else 'v1.1'
            revision=provenance.get(name,{}).get('revision') or resolve_revision(repo,ref)
            weight_name='biovil_t_image_model_proj_size_128.pt' if temporal else 'biovil_image_resnet50_proj_size_128.pt'
            weight=hf_hub_download(repo,weight_name,revision=revision,cache_dir=CACHE)
            model=ImageModel(img_encoder_type=ImageEncoderType.RESNET50_MULTI_IMAGE if temporal else ImageEncoderType.RESNET50,joint_feature_size=128)
            state=torch.load(weight,map_location='cpu',weights_only=True)
            model.load_state_dict(state,strict=True)
            del state
            model.eval().requires_grad_(False)
            transform=create_chest_xray_transform_for_inference(resize=512,center_crop_size=448 if temporal else 480)
            pooled=[]; projected=[]
            with torch.inference_mode():
                for start in range(0,len(files),4):
                    batch=torch.stack([transform(load_image(ROOT/'cache'/'images'/f)) for f in files[start:start+4]])
                    result=model(batch)
                    pooled.append(result.img_embedding.numpy())
                    projected.append(torch.nn.functional.normalize(result.projected_global_embedding,dim=-1).numpy())
                    print(f'{name}: {min(start+4,len(files))}/{len(files)} images',flush=True)
            arrays={'pooled':np.concatenate(pooled),'projected':np.concatenate(projected)}
            prep=f'official hi-ml-multimodal 0.2.2 loader minmax uint8 grayscale; resize short side 512; center crop {448 if temporal else 480}; ToTensor; repeat 3 channels; single image'
        elif name=='rad_dino':
            from transformers import AutoImageProcessor, AutoModel
            from PIL import Image
            repo='microsoft/rad-dino'
            revision=provenance.get(name,{}).get('revision') or resolve_revision(repo)
            processor=AutoImageProcessor.from_pretrained(repo,revision=revision,cache_dir=CACHE)
            model=AutoModel.from_pretrained(repo,revision=revision,cache_dir=CACHE,use_safetensors=True,attn_implementation='sdpa')
            model.eval().requires_grad_(False)
            weight=hf_hub_download(repo,'model.safetensors',revision=revision,cache_dir=CACHE)
            vectors=[]
            with torch.inference_mode():
                for start in range(0,len(files),2):
                    images=[Image.open(ROOT/'cache'/'images'/f).convert('RGB') for f in files[start:start+2]]
                    inputs=processor(images=images,return_tensors='pt')
                    result=model(**inputs)
                    vectors.append(result.last_hidden_state[:,0,:].numpy())
                    for image in images: image.close()
                    print(f'{name}: {min(start+2,len(files))}/{len(files)} images',flush=True)
            arrays={'cls':np.concatenate(vectors)}
            prep={'processor':processor.to_dict(),'feature':'final normalized CLS token; native checkpoint resolution; no target fine tuning'}
        else: raise ValueError(name)
        for value in arrays.values():
            assert value.shape[0]==94 and np.isfinite(value).all()
        np.savez_compressed(target,filenames=np.array(files),**arrays)
        provenance[name]={'repository':repo,'revision':revision,'weight_file':Path(weight).name,
                          'weight_sha256':sha(weight),'features_sha256':sha(target),'preprocessing':prep,
                          'shapes':{k:list(v.shape) for k,v in arrays.items()},'seconds':time.monotonic()-started,
                          'frozen':True,'labels_used_for_feature_extraction':False}
        provenance_path.write_text(json.dumps(provenance,indent=2),encoding='utf-8')
        print(json.dumps(provenance[name],default=str),flush=True)
        del model, arrays
        gc.collect()


if __name__=='__main__': main()
