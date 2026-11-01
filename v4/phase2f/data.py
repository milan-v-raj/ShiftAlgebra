"""Hash-audited, transport-neutral CIFAR-100 packing and loading."""
from __future__ import annotations
import hashlib,json,pickle
from pathlib import Path
import numpy as np
import torch
from PIL import Image
ROOT=Path(__file__).resolve().parents[2]
IMAGE_ROOT=ROOT/'data/external/cifar100-images'; PICKLE_ROOT=ROOT/'data/external/cifar100-test-pickle/data'; CACHE=ROOT/'data/cache/cifar100-fullvector.pt'
EXPECTED_TEST_MD5='f0ef6b0ae62326f3e7ffdfab6717acfc'
def digest(path,kind='sha256'):
 h=hashlib.new(kind)
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def source_commit(path):return __import__('subprocess').check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()
def prepare():
 test_pickle=PICKLE_ROOT/'test'; observed=digest(test_pickle,'md5')
 if observed!=EXPECTED_TEST_MD5:raise ValueError(f'canonical test MD5 mismatch: {observed}')
 meta=pickle.load((PICKLE_ROOT/'meta').open('rb'),encoding='bytes');names=[x.decode() for x in meta[b'fine_label_names']]
 official=pickle.load(test_pickle.open('rb'),encoding='bytes');ox=np.asarray(official[b'data'],dtype=np.uint8).reshape(-1,3,32,32);oy=np.asarray(official[b'fine_labels']);train=[];train_y=[];test=[];test_y=[];verified=0
 for label,name in enumerate(names):
  tr=sorted((IMAGE_ROOT/'train'/name).glob('*.png'));te=sorted((IMAGE_ROOT/'test'/name).glob('*.png'))
  if len(tr)!=500 or len(te)!=100:raise ValueError((name,len(tr),len(te)))
  train.extend(np.transpose(np.asarray(Image.open(p).convert('RGB'),dtype=np.uint8),(2,0,1)) for p in tr);train_y.extend([label]*len(tr))
  canonical={hashlib.sha256(ox[i].tobytes()).digest() for i in np.where(oy==label)[0]}
  for p in te:
   a=np.transpose(np.asarray(Image.open(p).convert('RGB'),dtype=np.uint8),(2,0,1));verified+=hashlib.sha256(a.tobytes()).digest() in canonical;test.append(a);test_y.append(label)
 if verified!=10000:raise ValueError(f'only {verified}/10000 test images match canonical pixels')
 provenance={'image_transport':'https://github.com/cyizhuo/CIFAR-100-dataset','image_transport_commit':source_commit(IMAGE_ROOT),'test_pickle_transport':'https://github.com/DevFelas/cifar-100-python','test_pickle_transport_commit':source_commit(PICKLE_ROOT.parent.parent),'canonical_test_md5':observed,'canonical_test_pixel_matches':verified,'class_names':names,'train_count':50000,'test_count':10000,'note':'GitHub repositories are transport only; dataset files are not redistributed.'}
 CACHE.parent.mkdir(parents=True,exist_ok=True);torch.save({'train_images':torch.from_numpy(np.stack(train)),'train_labels':torch.tensor(train_y),'test_images':torch.from_numpy(np.stack(test)),'test_labels':torch.tensor(test_y),'provenance':provenance},CACHE);(ROOT/'v4/phase2f/DATA_PROVENANCE.json').write_text(json.dumps(provenance,indent=2,sort_keys=True)+'\n');print(CACHE);return provenance
def load():
 if not CACHE.exists():prepare()
 return torch.load(CACHE,map_location='cpu',weights_only=False)
def balanced_indices(labels,per_class,offset):
 out=[]
 for c in range(100):
  ix=torch.nonzero(labels==c).flatten()
  if offset+per_class>len(ix):raise ValueError(c)
  out.append(ix[offset:offset+per_class])
 return torch.cat(out)
if __name__=='__main__':prepare()
