## standard lib
from pathlib import Path
from time import time
from types import SimpleNamespace as SN
import json
import pickle
import shutil
from textwrap import dedent
from itertools import product
import sys
import ast
import csv

## standard scipy
import ipdb
import matplotlib
import numpy as np
from numpy import array, exp, zeros, maximum, indices
def ceil(x): return np.ceil(x).astype(int)
def floor(x): return np.floor(x).astype(int)
import torch

import networkx as nx

from matplotlib                import pyplot as plt
from scipy.ndimage             import label as labelComponents
from scipy.ndimage             import zoom
from skimage.feature           import peak_local_max
from skimage.io                import imsave
from skimage.measure           import regionprops
from skimage.morphology.binary import binary_dilation
from skimage.segmentation      import find_boundaries
from tifffile                  import imread

# import isbidata

## 3rd party 
# import augmend

## local
from torch_models import Unet3, init_weights, nn
from pointmatch import snnMatch
import tracking2

"""
RUN ME ON SLURM!!
sbatch -J e23-mau -p gpu --gres gpu:1 -n 1 -t  6:00:00 -c 1 --mem 128000 -o slurm_out/e23-mau.out -e slurm_err/e23-mau.out --wrap '/bin/time -v python e23_mauricio2.py'
"""

def load_tif(name): return imread(name) 
def load_pkl(name): 
  with open(name,'rb') as file:
    return pickle.load(file)
def save_pkl(name, stuff):
  with open(name,'wb') as file:
    pickle.dump(stuff, file)
def save_png(name, img):
  imsave(name, img, check_contrast=False)
def save_tif(name, img):
  imsave(name, img, check_contrast=False)

def wipedir(path):
  path = Path(path)
  if path.exists(): shutil.rmtree(path)
  path.mkdir(parents=True, exist_ok=True)

"""
UTILITIES
"""

def createTarget(pts, target_shape, sigmas):
  s  = np.array(sigmas)
  ks = floor(7*s).astype(int)   ## extend support to 7/2 sigma in every direc
  ks = ks - ks%2 + 1            ## enfore ODD shape so kernel is centered! 

  pts = np.array(pts).astype(int)

  ## create a single Gaussian kernel array
  def f(x):
    x = x - (ks-1)/2
    return exp(-(x*x/s/s).sum()/2)
  kern = array([f(x) for x in indices(ks).reshape((len(ks),-1)).T]).reshape(ks)
  
  target = zeros(ks + target_shape) ## include border padding
  w = ks//2                         ## center coordinate of kernel
  pts_offset = pts + w              ## offset by padding

  for p in pts_offset:
    target_slice = tuple(slice(a,b+1) for a,b in zip(p-w,p+w))
    target[target_slice] = maximum(target[target_slice], kern)

  remove_pad = tuple(slice(a,a+b) for a,b in zip(w,target_shape))
  target = target[remove_pad]

  return target

def splitIntoPatches(img_shape, outer_shape=(256,256), min_border_shape=(24,24), divisor=(8,8)):
  """
  Split image into non-overlapping `inner` rectangles that exactly cover
  `img_shape`. Grow these rectangles by `border_shape` to produce overlapping
  `outer` rectangles to provide context to all `inner` pixels. These borders
  should be half the receptive field width of our CNN.

  CONSTRAINTS
  outer_shape % divisor == 0
  img_shape   >= outer_shape 
  outer_shape > 2*min_border_shape

  GUARANTEES
  outer_shape = constant across patches
  inner_shape <= outer_shape - 2 * min_border_shape

  borders are evenly distributed on all sides, except on image boundaries
  max(inner_shapes across patches) - min(inner_shapes across patches) <= 1 forall shapes
  
  """
  
  img_shape = array(img_shape)
  divisor = array(divisor)
  min_border_shape = array(min_border_shape)

  ## This means we'll have one image with almost all the content in the mask, and 2^d - 1 other images with almost no content.
  ## One alternative is to rescale images to a nice multiple of divisor? Or pad them ? 
  ## We could rely on this !? Small images are resized 
  # patchmax = floor(img_shape/divisor)*divisor 

  outer_shape = array(outer_shape).clip(max=img_shape)
  ## we just enforced this...
  assert all(img_shape>=outer_shape), f"Error: `outer_shape` doesn't fit. inner: {img_shape}, outer: {outer_shape} ..."
  ## outer_shape % divisor == 0 (REQUIRED)
  ## Setting outer_shape.max = img_shape ensures that img_shape % 8 == 0 if img_shape < outer_shape
  assert all(outer_shape % divisor == 0), f"img_shape ({img_shape}) not divisible by ({divisor})"

  assert all(outer_shape>=2*min_border_shape), f"Error: borders too wide: {outer_shape} < 2*{min_border_shape}"


  # our actual shape will be <= this desired shape. `outer_shape` is fixed,
  # but the border shape will grow.
  desired_inner_shape = outer_shape - min_border_shape

  ## round up, shrinking the actual inner_shape
  inner_counts = ceil(img_shape / desired_inner_shape)
  inner_shape_float = img_shape / inner_counts

  ## "Recomputed Outer Shape"
  ros = ceil(inner_shape_float) + min_border_shape
  ros = ceil(ros/divisor) * divisor

  ## this function runs per-dimension n
  ## i : ith patch along dimension 
  ## n : dimension index i.e. one of 0,1[,2]
  def f(i,n):
    a = inner_shape_float[n]
    b = floor(a*i)      # slice start
    c = floor(a*(i+1))  # slice stop
    inner = slice(b,c)
    
    w = c-b

    # shift `outer` patch inwards when we hit a border to maintain outer_shape.
    if b==0:                          # left border
      outer = slice(0,ros[n])
      inner_rel = slice(0,w)
    elif c==img_shape[n]:             # right border 
      outer = slice(img_shape[n]-ros[n],img_shape[n])
      inner_rel = slice(ros[n]-w,ros[n])
    else:
      r = b - min_border_shape[n]
      outer = slice(r,r+ros[n])
      inner_rel = slice(min_border_shape[n],min_border_shape[n]+w)
    return SN(inner=inner, outer=outer, inner_rel=inner_rel)

  ndim = len(inner_counts)
  slices_lists = [[f(i,n) for i in range(inner_counts[n])] 
                          for n in range(ndim)]

  def g(s):
    inner = tuple(x.inner for x in s)
    outer = tuple(x.outer for x in s)
    inner_rel = tuple(x.inner_rel for x in s)
    return SN(inner=inner, outer=outer, inner_rel=inner_rel)

  # equivalent to itertools.product(*slices_lists)
  # prod = array(np.meshgrid(*slices_lists)).reshape((ndim,-1)).T

  res = [g(s) for s in product(*slices_lists)]
  return res

## rescale pts to be consistent with scipy.ndimage.zoom(img,scale)
def zoom_pts(pts,scale):
  # assert type(pts) is np.ndarray
  pts = pts+0.5                         ## move origin from middle of first bin to left boundary (array index convention)
  pts = pts * scale                     ## rescale
  pts = pts-0.5                         ## move origin back to middle of first bin
  pts = np.round(pts).astype(np.uint32) ## binning
  return pts

## x : image::ndarray
## kind : in ['I','L'] for "Intensity" / "Label"
## colors : colormap
def img2png(x, kind, colors=None, greynorm=True):
  
  assert kind in ['I','L']
  if 'float' in str(x.dtype): assert kind=='I'
  elif 'int' in str(x.dtype): assert kind=='L'
  else: assert False, f'kind {kind} does not match image type {x.dtype}.'

  if 'float' in str(x.dtype) and colors:
    # assert type(colors) is matplotlib.colors.ListedColormap
    cmap = colors
  elif 'float' in str(x.dtype) and not colors:
    # assert type(colors) is matplotlib.colors.ListedColormap
    cmap = plt.cm.gray
  elif 'int' in str(x.dtype) and type(colors) is list:
    cmap = np.array([(0,0,0)] + colors*256)[:256]
    cmap = matplotlib.colors.ListedColormap(cmap)
  elif 'int' in str(x.dtype) and type(colors) is matplotlib.colors.ListedColormap:
    cmap = colors
  elif 'int' in str(x.dtype) and colors is None:
    cmap = np.random.rand(256,3).clip(min=0.2)
    cmap[0] = (0,0,0)
    cmap = matplotlib.colors.ListedColormap(cmap)

  def _colorseg(seg):
    m = seg!=0
    seg[m] %= 255 ## we need to save a color for black==0
    seg[seg==0] = 255
    seg[~m] = 0
    rgb = cmap(seg)
    return rgb

  _dtype = x.dtype
  nd = x.ndim

  if nd==3:
    a,b,c = x.shape
    yx = x.max(0)
    zx = x.max(1)
    zy = x.max(2)
    x0 = np.zeros((a,a), dtype=x.dtype)
    x  = np.zeros((b+a+1,c+a+1), dtype=x.dtype)
    x[:b,:c] = yx
    x[b+1:,:c] = zx
    x[:b,c+1:] = zy.T
    x[b+1:,c+1:] = x0

  assert x.dtype == _dtype

  # ipdb.set_trace()

  if 'int' in str(x.dtype):
    x = _colorseg(x)
  else:
    if greynorm:
      x = norm_minmax01(x)
    x = cmap(x)
  
  x = (x*255).astype(np.uint8)

  if nd==3:
    x[b,:] = 255 # white line
    x[:,c] = 255 # white line

  return x

def norm_minmax01(x):
  hi = x.max()
  lo = x.min()
  if hi==lo: 
    return x-lo
  else: 
    return (x-lo)/(hi-lo)

def norm_percentile01(x,p0,p1):
  lo,hi = np.percentile(x,[p0,p1])
  if hi==lo: 
    return x-hi
  else: 
    return (x-lo)/(hi-lo)

## ONLY pad on the right ends, so `pts` location is still valid.
def pad_until_divisible(raw, patch_size, divisor, return_pad=False):
  rawsize = array(raw.shape)
  patch_size = array(patch_size)
  divisor = array(divisor)
  desired_rawsize = ceil(rawsize/divisor)*divisor
  padding = np.where(rawsize < patch_size, desired_rawsize - rawsize, 0)
  raw = np.pad(raw, [(0,p) for p in padding], constant_values=0)
  if return_pad: return raw, padding
  return raw

## don't use pandas `read_csv()` for parsing! use python's `ast.literal_eval()`
## WARNING: We attempt to interpret all cells in the table as python, and only 
## fall back to `str` on failure. If we WANT a string, we have to wrap it in extra quotes,
## or ensure that `literal_eval()` fails.
def load_isbi_csv(isbiname):

  def parse(x):
    try: 
      y = ast.literal_eval(x)
      if type(y) in [int,float,tuple,list,bool]: x=y
    except:
      pass
    return x

  for row in csv.DictReader(open('isbi-stats.csv','r')):
    if row['name']==isbiname:
      return {k:parse(v) for k,v in row.items()}

"""
Core Functions
"""




## Parameters for data(), train(), and predict()
def params(isbiname = "Fluo-C2DL-Huh7"):

  savedir = Path(f"cpnet-out/{isbiname}/")
  savedir.mkdir(parents=True,exist_ok=True)
  # base = f"/projects/project-broaddus/rawdata/isbi_train/{isbiname}/"
  base = f"data-isbi/{isbiname}/"

  isbi = load_isbi_csv(isbiname)

  ####### data, train, predict #######

  PR = SN()
  PR.isbiname = isbiname
  PR.savedir = savedir
  PR.ndim = 2 if "2D" in isbiname else 3

  tname = isbi['tname']
  if tname==3:
    PR.name_raw = base + "{dset}/t{time:03d}.tif"
    PR.name_pts = base + "{dset}_GT/TRA/man_track{time:03d}.tif"
  elif tname==4:
    PR.name_raw = base + "{dset}/t{time:04d}.tif"
    PR.name_pts = base + "{dset}_GT/TRA/man_track{time:04d}.tif"

  tb = isbi['times 01']
  subsample = isbi['take nth']
  alldata = np.array([dict(dset=d, time=t)
                        for d in ['01']
                        for t in range(tb[0], tb[1], subsample)])
  np.random.seed(42)
  np.random.shuffle(alldata)
  PR.trainvalidata = alldata

  tb = isbi['times 02']
  PR.preddata = np.array([dict(dset=d, time=t)
                        for d in ['02']
                        for t in range(tb[0], tb[0]+8)])
  
  ## predict
  PR.mode = 'NoGT' ## 'withGT'

  PR.aniso = np.array(isbi['scales'])
  if PR.ndim==3: PR.aniso = PR.aniso / PR.aniso[2]

  ## train
  cmap = np.zeros((256,3),np.float32) #np.random.rand(256,3).clip(min=0.2)
  cmap[0] = (0,0,0) # background
  cmap[1] = (0,0,1) # prediction
  cmap[2] = (0,1,0) # ground truth
  cmap[3] = (1,0,0) # prediction + ground truth
  cmap = matplotlib.colors.ListedColormap(cmap)
  PR.cmap_glance = cmap

  ## DEFAULTS

  if PR.ndim==2:
    ## data, predict
    PR.divisor = (8,8)
    PR.outer_shape = (128,128)
    ## train
    # PR.sigma = (5,5)
    # PR.border = [0,0]
    # PR.zoom = (0.5,0.5)
  if PR.ndim==3:
    ## data, predict
    PR.outer_shape = (16,128,128)
    PR.divisor = (1,8,8)
    ## train
    # PR.sigma = (3,5,5)
    # PR.zoom = (1,0.5,0.5)
    # PR.border = [0,0,0]

  PR.zoom = isbi['zoom']
  PR.sigma = isbi['sigma']
  PR.sparse = isbi['sparse']
  
  ## functions shared across data(), train(), and predict()

  if PR.ndim==2:
    ## data, predict
    PR.splitIntoPatches = lambda x: splitIntoPatches(x, outer_shape=PR.outer_shape, min_border_shape=(16,16), divisor=PR.divisor)
    PR.splitIntoPatchesPred = PR.splitIntoPatches
    ## train, predict
    PR.findPeaks = lambda x: peak_local_max(x, threshold_abs=.5, exclude_border=False, footprint=np.ones([5,5]))
    PR.snnMatch  = lambda yt, y: snnMatch(yt, y, dub=10, scale=[1,1]) ## y_true, y_predicted
    PR.buildUNet = lambda : Unet3(16, [[1],[1]], pool=(2,2), kernsize=(5,5), finallayer=nn.Sequential)
  if PR.ndim==3:
    ## data, predict
    PR.splitIntoPatches = lambda x: splitIntoPatches(x, outer_shape=PR.outer_shape, min_border_shape=(0,16,16), divisor=PR.divisor)
    PR.splitIntoPatchesPred = PR.splitIntoPatches
    ## train, predict
    PR.snnMatch  = lambda yt, y: snnMatch(yt, y, dub=10, scale=PR.aniso)
    PR.buildUNet = lambda : Unet3(16, [[1],[1]], pool=(1,2,2), kernsize=(3,5,5), finallayer=nn.Sequential)
    PR.findPeaks = lambda x: peak_local_max(x, threshold_abs=.5, exclude_border=False, footprint=np.ones([3,5,5]))
  
  return PR


## Tiling patches with overlapping borders. Requires loss masking.
## Const outer size % 8 = 0.
def data(PR):

  def f(dikt):
    raw = load_tif(PR.name_raw.format(**dikt)).astype(np.float32) ## cast from isbi data in u8 or u16  #.transpose([1,0,2,3])
    lab = load_tif(PR.name_pts.format(**dikt))
    pts = np.array([x['centroid'] for x in regionprops(lab)])
    print("rawshape 1: ", raw.shape)
    raw = zoom(raw, PR.zoom, order=1)
    print("rawshape 2: ", raw.shape)
    raw = pad_until_divisible(raw, PR.outer_shape, PR.divisor)
    print("rawshape 3: ", raw.shape)
    # lab = zoom(lab, PR.zoom, order=1)
    pts = zoom_pts(pts, PR.zoom)
    ## cast sizes to reduce dataset size 4x
    raw = norm_percentile01(raw,2,99.4).astype(np.float16)
    target = createTarget(pts, raw.shape, PR.sigma).astype(np.float16)
    patches = PR.splitIntoPatches(raw.shape)
    raw_patches = [raw[p.outer] for p in patches]
    target_patches = [target[p.outer] for p in patches]
    samples = [SN(raw=r, target=t, inner=p.inner, outer=p.outer, inner_rel=p.inner_rel, **dikt) 
                  for r,t,p in zip(raw_patches,target_patches,patches)]
    return samples

  # return pickle.load(open(str(PR.savedir / 'data/filtered.pkl'), 'rb'))

  data = [f(dikt) for dikt in PR.trainvalidata]
  data = [s for dat in data for s in dat]

  if PR.sparse:
    data = [sample for sample in data if sample.target.max()>0.5]
    # empty = np.array([sample for sample in data if sample.target.max()<0.5])
    # np.random.shuffle(empty)
    # empty = empty[:len(anno)]
    # data = anno + list(empty)

  # ipdb.set_trace()

  wipedir(PR.savedir/"data/")
  save_pkl(PR.savedir/"data/dataset.pkl", data)

  ## save train/vali/test data
  wipedir(PR.savedir/"data/png/")
  ids = ceil(np.linspace(0,len(data)-1,10)) if len(data)>10 else range(len(data)) ## <= 10 evenly sampled patches
  for i in ids:
    s = data[i]
    r = img2png(s.raw, 'I', greynorm=False)
    mask = find_boundaries(s.target>0.5, mode='inner')
    t = img2png(mask.astype(np.uint8), 'L', colors=PR.cmap_glance) ## just use any label cmap
    composite = r.copy()
    composite[mask] = (r[mask]/2.0 + t[mask]/2.0).astype(np.uint8).clip(min=0,max=255)
    imsave(PR.savedir/f'data/png/t{s.time:03d}-d{i:04d}.png', composite)

  return data


## NOTE: train() includes additional data filtering.
def train(PR, continue_training=False):

  CONTINUE = continue_training
  print("CONTINUE ? : ", bool(CONTINUE))

  print(f"""
    Begin training CP-Net on {PR.isbiname}
    Savedir is {PR.savedir / "train"}
    """)

  dataset = load_pkl(PR.savedir/"data/dataset.pkl")
  dataset = np.array(dataset)

  ## network, weights and optimization
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  print(f"We're using torch device {device} .")
  net = PR.buildUNet()
  net = net.to(device)
  init_weights(net)
  opt = torch.optim.Adam(net.parameters(), lr = 1e-4)

  ## load weights and sample train/vali assignment from disk?
  if CONTINUE:
    labels = load_pkl(PR.savedir / "train/labels.pkl")
    net.load_state_dict(torch.load(PR.savedir / f'train/m/best_weights_latest.pt', map_location=torch.device(device)))
    history = load_pkl(PR.savedir / 'train/history.pkl')
  else:
    wipedir(PR.savedir/'train/m')
    wipedir(PR.savedir/"train/glance_output_train/")
    wipedir(PR.savedir/"train/glance_output_vali/")
    N = len(dataset)
    # a, b = (N*5)//8, (N*7)//8  ## MYPARAM train / vali / test 
    a = (N*7)//8 ## don't use test patches. test on full images.
    labels = np.zeros(N,dtype=np.uint8)
    labels[a:]=1; ## labels[b:]=2 ## 0=train 1=vali 2=test
    np.random.shuffle(labels)
    save_pkl(PR.savedir / "train/labels.pkl", labels)
    history = SN(lossmeans=[], valimeans=[])

  assert len(dataset)>8
  assert len(labels)==len(dataset)

  ## add loss weights to each patch
  for s in dataset:
    s.weights = np.zeros(s.target.shape)
    s.weights[s.inner_rel] = 1
    if PR.sparse:
      s.weights = (s.target > np.exp(-0.5*(3**2))).astype(np.float32)

    # s.weights = binary_dilation(s.target>0 , np.ones((1,7,7)))
    # s.weights = (s.target > 0)
    # print("{:.3f}".format(s.weights.mean()),end="  ")

  ## augmentation by flips and rotations
  def augmentSample(sample):
    s = sample
    
    # rotate -90 + flip x
    if np.random.rand() < 0.5:
      for x in [s.raw, s.target, s.weights]:
        x = x.transpose([1,0]) if PR.ndim==2 else x.transpose([0,2,1])
    
    # flip y[z]
    if np.random.rand() < 0.5:
      for x in [s.raw, s.target, s.weights]:
        x = x[::-1]
    
    # flip x[y]
    if np.random.rand() < 0.5:
      for x in [s.raw, s.target, s.weights]:
        x = x[:, ::-1]

    # flip [x] if dim==3
    if np.random.rand() < 0.5 and PR.ndim==3:
      for x in [s.raw, s.target, s.weights]:
        x = x[:, :, ::-1]


  ## Split train/vali
  traindata = dataset[labels==0] # & (tmax > 0.99) 
  validata  = dataset[labels==1] # & (tmax > 0.99) 

  N_train = len(traindata)
  N_vali  = len(validata)
  print(f"""
    Data filtered from N={len(dataset)} to 
    N_train={N_train} , N_vali={N_vali}
    """)

  def mse_loss(s, augment=True, mode=None):
    assert mode in ['train', 'vali', 'glance']

    x  = s.raw.copy()
    yt = s.target.copy()
    w  = s.weights.copy()

    if augment: augmentSample(s)

    x  = torch.from_numpy(x.copy() ).float().to(device, non_blocking=True)
    yt = torch.from_numpy(yt.copy()).float().to(device, non_blocking=True)
    w  = torch.from_numpy(w.copy() ).float().to(device, non_blocking=True)

    if mode == 'train':
      y  = net(x[None,None])[0,0]
    elif mode in ['vali','glance']:
      with torch.no_grad(): 
        y  = net(x[None,None])[0,0]

    ## only apply loss on non-overlapping component
    y  = y[s.inner_rel]
    x  = x[s.inner_rel]
    yt = yt[s.inner_rel]
    w  = w[s.inner_rel]
    
    loss = torch.abs((w*(y-yt)**2)).mean()

    if mode=='train':
      loss.backward()
      opt.step()
      opt.zero_grad()
      y = y.detach().cpu().numpy()
      loss = float(loss.detach().cpu())
      return y,loss

    if mode=='vali':
      y = y.cpu().numpy()
      loss = float(loss.detach().cpu())
      pts = PR.findPeaks(y)
      gt_pts = PR.findPeaks(yt.detach().cpu().numpy().astype(np.float32))
      
      matching = PR.snnMatch(gt_pts, pts)
      scores = SN(loss=loss, f1=matching.f1, height=y.max())
      return y, scores

    if mode=='glance':
      x = x.cpu().numpy()
      y = y.cpu().numpy()
      yt = yt.cpu().numpy()
      w = w.cpu().numpy()

      r = img2png(x, 'I')
      p = img2png(y, 'I', colors=plt.cm.magma)
      w = img2png(w, 'I')
      # t = img2png((yt > 0.9).cpu().numpy().astype(np.uint8), 'L')
      pts = PR.findPeaks(y)
      gt_pts = PR.findPeaks(yt.astype(np.float32))
      t = np.zeros(x.shape,np.uint8)
      t[tuple(gt_pts.T)] = 1
      t[tuple(pts.T)] += 2 ## gives 3 on overlap
      # ipdb.set_trace()

      t = img2png(t, 'L', colors=PR.cmap_glance)
      composite = np.round(r/2 + p/2).astype(np.uint8).clip(min=0,max=255)
      ## t is sparse. set composite to t only where t has nonzero value
      m = np.any(t[:,:,:3]!=0 , axis=2)
      composite[m] = t[m]
      return composite

  def trainOneEpoch():
    _losses = []
    idxs = np.arange(N_train)
    np.random.shuffle(idxs)
    tic = time()
    for i in range(N_train):
      s  = traindata[idxs[i]]
      y,loss = mse_loss(s, augment=True, mode='train')
      _losses.append(loss)
      dt = time()-tic; tic = time()
      print(f"it {i}/{N_train}, dt {dt:5f}, max {y.max():5f}", end='\r',flush=True)

    history.lossmeans.append(np.nanmean(_losses))

  def validateOneEpoch():
    _valiscores = []
    idxs = np.arange(N_vali)
    np.random.shuffle(idxs)
    for i in idxs:
      s = validata[i]
      y, scores = mse_loss(s,augment=False,mode='vali')
      _valiscores.append((scores.loss, scores.f1, scores.height))
      # if i%10==0: print(f"_scores",_scores, end='\n',flush=True)

    history.valimeans.append(np.nanmean(_valiscores,axis=0))

    torch.save(net.state_dict(), PR.savedir / f'train/m/best_weights_latest.pt')

    valikeys   = ['loss','f1','height']
    valiinvert = [1,-1,-1] # minimize, maximize, maximize
    valis = np.array(history.valimeans).reshape([-1,3])*valiinvert

    for i,k in enumerate(valikeys):
      if np.nanmin(valis[:,i])==valis[-1,i]:
        torch.save(net.state_dict(), PR.savedir / f'train/m/best_weights_{k}.pt')

  def predGlances(time):
    ids = [0,N_train//2,N_train-1]
    for i in ids:
      composite = mse_loss(traindata[i], augment=True, mode='glance')
      save_png(PR.savedir/f'train/glance_output_train/a_{i:04d}_{time:03d}.png', composite)

    ids = [0,N_vali//2,N_vali-1]
    for i in ids:
      composite = mse_loss(validata[i], augment=False, mode='glance')
      save_png(PR.savedir/f'train/glance_output_vali/a_{i:04d}_{time:03d}.png', composite)

  n_pix = np.sum([np.prod(d.raw.shape) for d in traindata]) / 1_000_000 ## Megapixels of raw data in traindata
  if PR.ndim==2:
    rate = 1.287871 if str(device)!='cpu' else 0.074418 # updated for M1. Old mac rate: 0.0435 [megapixels / sec] (1.4 was old gpu rate... did i slow down?)
  elif PR.ndim==3:
    rate = 0.976863 if str(device)!='cpu' else 0.0310 # [megapixels / sec] (1.0 was old gpu rate... did i slow down?)
  N_epochs=300 ## MYPARAM

  print(f"Estimated Time: {n_pix} Mpix / {rate} Mpix/s = {n_pix/rate/60*N_epochs:.2f}m = {300*n_pix/60/60/rate:.2f}h \n")
  print(f"\nBegin training for {N_epochs} epochs...\n\n")

  for ep in range(N_epochs):
    tic = time()
    trainOneEpoch()
    validateOneEpoch()
    save_pkl(PR.savedir / "train/history.pkl", history)
    if ep in range(10) or ep%10==0: predGlances(ep)
    dt  = time() - tic

    print("\033[F",end='') ## move cursor UP one line 
    print(f"finished epoch {ep+1}/{N_epochs}, loss={history.lossmeans[-1]:4f}, dt={dt:4f}, rate={n_pix/dt:5f} Mpix/s", end='\n',flush=True)


## Make predictions for each saved weight set : 'latest','loss','f1','height'
## Include avg/min across predictions too! Simple model ensembling.
def predict(PR):

  wipedir(PR.savedir / "predict")

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  net = PR.buildUNet()
  net = net.to(device)

  def predsingle(dikt):
    raw = load_tif(PR.name_raw.format(**dikt)) #.transpose([1,0,2,3])[1]
    print("rawshape 1: ", raw.shape)
    raw = zoom(raw, PR.zoom, order=1)
    print("rawshape 2: ", raw.shape)
    raw, padding = pad_until_divisible(raw, PR.outer_shape, PR.divisor, return_pad=True)
    print("rawshape 3: ", raw.shape)
    raw = norm_percentile01(raw,2,99.4)

    if PR.mode=='withGT':
      lab = load_tif(PR.name_pts.format(**dikt)) #.transpose([])
      pts = np.array([x['centroid'] for x in regionprops(lab)])
      gtpts = zoom_pts(np.array(gtpts) , PR.zoom) #.astype(np.int)

    ## seamless prediction tiling
    pred = np.zeros(raw.shape)
    for p in PR.splitIntoPatchesPred(pred.shape):
      x = torch.Tensor(raw[p.outer][None,None]).to(device)
      with torch.no_grad():
        pred[p.inner] = net(x).cpu().numpy()[0,0][p.inner_rel]
    ss = tuple([slice(0,s-p) for s,p in zip(pred.shape, padding)])
    pred = pred[ss]

    ## find and scale peaks back to orig space
    height = pred.max()    
    pts = PR.findPeaks(pred)
    pts = zoom_pts(pts , 1 / np.array(PR.zoom))
    
    if PR.mode=='withGT':
      matching = PR.snnMatch(gtpts, pts)
      print(dedent(f"""
          weights : {weights}
             time : {time:03d}
               f1 : {matching.f1:.3f}
        precision : {matching.precision:.3f}
           recall : {matching.recall:.3f}
        """))

    def f():
      r = img2png(raw, 'I')
      p = img2png(labelComponents(pred>0.5)[0], 'L') #, colors=plt.cm.magma)
      composite = np.round(r/2 + p/2).astype(np.uint8).clip(min=0,max=255)
      # ipdb.set_trace()
      # t = img2png((yt > 0.9).numpy().astype(np.uint8))
      # m = np.any(t[:,:,:3]!=0 , axis=2)
      # composite[m] = t[m]
      return composite
    # composite = f()

    return SN(**locals())

  N_imgs = len(PR.preddata)

  ltps = []
  for weights in ['f1']: #['latest','loss','f1','height']:
    net.load_state_dict(torch.load(PR.savedir / f'train/m/best_weights_{weights}.pt', map_location=torch.device(device)))

    for i, dikt in enumerate(PR.preddata):
      print(f"Predicting on image {i+1}/{N_imgs}...", end='\r',flush=True)
      d = predsingle(dikt)
      ltps.append(d.pts)
      # save_png(PR.savedir/"predict/t{time:04d}-{weights}.png".format(**dikt,weights=weights), d.composite)

  print(f"Run tracking...", end='\n', flush=True)
  rawshape = load_tif(PR.name_raw.format(**PR.preddata[0])).shape
  # track_labeled_images = tracking.makeISBILabels(ltps,rawshape)
  # list_of_edges, list_of_labels = trackAndLabel(ltps)
  tb = tracking2.nn_tracking(ltps, aniso=PR.aniso)

  # tracking2.draw(tb)
  # plt.ion()
  # plt.show()
  # input()

  ## random colormap for tracking
  cmap = np.random.rand(256,3).clip(min=0.2)
  cmap[0] = (0,0,0)
  cmap = matplotlib.colors.ListedColormap(cmap)

  wipedir(PR.savedir/"track/png")
  for i, dikt in enumerate(PR.preddata):
    print(f"Saving image {i+1}/{N_imgs}...", end='\r',flush=True)
    rawpng = img2png(load_tif(PR.name_raw.format(**dikt)).astype(np.float32), 'I')
    # lab = tracking2.make_ISBI_label_img(tb,i,rawshape,halfwidth=6)
    lab = tracking2.createTarget(tb, i, rawshape, PR.sigma) ## WARNING: Using index `i` instead of dikt['time']
    labpng = img2png(lab, 'L', colors=cmap)
    composite = np.round(rawpng/2 + labpng/2).astype(np.uint8).clip(min=0,max=255)
    # save_tif(PR.savedir/"track/tif/img{time:03d}.tif".format(**dikt), track_labeled_images[i])
    # save_png(PR.savedir/"track/c{time:03d}.png".format(**dikt), composite)
    # save_png(PR.savedir/"track/r{time:03d}.png".format(**dikt), rawpng)
    save_png(PR.savedir/"track/png/img{time:03d}.png".format(**dikt), composite)


if __name__=="__main__":

  isbiname = sys.argv[1]
  PR = params(isbiname)

  DTP = sys.argv[2]

  if 'D' in DTP:
    dataset = data(PR)
  if 'Tc' in DTP:
    train(PR, continue_training=1)
  elif 'T' in DTP:
    train(PR, continue_training=0)
  if 'P' in DTP:
    predict(PR)





