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
import os

## standard scipy
import ipdb
import matplotlib
import numpy as np
from numpy import array, exp, zeros, maximum, indices
def ceil(x): return np.ceil(x).astype(int)
def floor(x): return np.floor(x).astype(int)
def round(x): return np.round(x).astype(int)
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

import localinfo

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
  s  = array(sigmas)
  ks = floor(7*s).astype(int)   ## extend support to 7/2 sigma in every direc
  ks = ks - ks%2 + 1            ## enfore ODD shape so kernel is centered! 

  pts = array(pts).astype(int)

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

## Build a list of slices that can tile an image into potentially overlapping
## (outer) patches, but that also contain a NON-overlapping (inner) subregion.
def splitIntoPatches(img_shape, desired_outer_shape, min_border_shape, divisor):
  """
  Split image into non-overlapping `inner` rectangles that exactly cover
  `img_shape`. Grow these rectangles by `border_shape` to produce overlapping
  `outer` rectangles to provide context to all `inner` pixels. We must mask the loss
  to only the inner region, otherwise we mix train and test data.

  All of inner, outer, and inner_rel are evenly divisible by `divisor`.
  """

  divisor = array(divisor)

  assert all(img_shape % divisor == 0)
  assert all(min_border_shape % divisor == 0)
  assert all(desired_outer_shape % divisor == 0)

  img_shape_blocks = array(img_shape) // divisor
  min_border_shape_blocks = array(min_border_shape) // divisor
  desired_outer_shape_blocks = array(desired_outer_shape) // divisor

  print(f"""
  img_shape = {img_shape}
  desired_outer_shape = {desired_outer_shape}
  min_border_shape = {min_border_shape}
  divisor = {divisor}
  img_shape_blocks = {img_shape_blocks}
  min_border_shape_blocks = {min_border_shape_blocks}
  desired_outer_shape_blocks = {desired_outer_shape_blocks}
  """
  )

  ## n : dimension index i.e. one of 0,1[,2]
  def singledim(n):
    b_img = img_shape_blocks[n]
    b_outer = desired_outer_shape_blocks[n]
    b_border = min_border_shape_blocks[n]

    if b_outer>=b_img:
      print(f"n={n}, n_patches = 1")
      return [SN(inner=slice(0,b_img), outer=slice(0,b_img), inner_rel=slice(0,b_img))]

    ## otherwise we need to split up the dimension

    b_max_inner = b_outer - 2 * b_border
    assert(b_max_inner > 0)
    n_patches = ceil(b_img/b_max_inner) ## this should be sufficient
    print(f"n={n}, n_patches = {n_patches}")
    # n_patches = max(ceil(b_img/b_max_inner),2)
    inner_patch_borders = round(np.linspace(0,b_img,n_patches+1))
    inner_start = inner_patch_borders[:-1]
    inner_stop  = inner_patch_borders[1:]
    outer_start = (inner_start - b_border).clip(min=0)
    outer_stop  = (inner_stop + b_border).clip(max=b_img)
    slices = [SN(inner=slice(inner_start[i],inner_stop[i]),
                  outer=slice(outer_start[i],outer_stop[i]),
                  inner_rel=slice(inner_start[i]-outer_start[i], inner_stop[i]-outer_start[i]),)
                  for i in range(n_patches)]
    return slices
  
  def transpose(s):
    inner = tuple(x.inner for x in s)
    outer = tuple(x.outer for x in s)
    inner_rel = tuple(x.inner_rel for x in s)
    return SN(inner=inner, outer=outer, inner_rel=inner_rel)

  samples = [transpose(s) for s in product(*[singledim(n) for n in range(len(img_shape))])]

  for sam in samples:
    sam.inner     = tuple(slice(s.start*divisor[n], s.stop*divisor[n]) for n,s in enumerate(sam.inner))
    sam.outer     = tuple(slice(s.start*divisor[n], s.stop*divisor[n]) for n,s in enumerate(sam.outer))
    sam.inner_rel = tuple(slice(s.start*divisor[n], s.stop*divisor[n]) for n,s in enumerate(sam.inner_rel))

  return samples


## rescale pts to be consistent with scipy.ndimage.zoom(img,scale)
def zoom_pts(pts,scale):
  # assert type(pts) is np.ndarray
  pts = pts+0.5                         ## move origin from middle of first bin to left boundary (array index convention)
  pts = pts * scale                     ## rescale
  pts = pts-0.5                         ## move origin back to middle of first bin
  pts = np.round(pts).astype(np.uint32) ## binning
  return pts

## guarantees that result is divisible by PR.divisor
## return exact zoom value so detections can be re-scaled properly
def zoom_img_make_divisible(raw,zoomtuple,divisor):
  z = zoomtuple
  s = array(raw.shape)
  z2 = ceil(s*z/divisor)*divisor/s
  raw = zoom(raw, z2, order=1)
  print(raw.shape, z2)
  assert all(array(raw.shape) % divisor == 0)
  return raw, z2

## guarantees that result is divisible by PR.divisor in every dimension for any size
## return padding value for prediction so it can be removed
## padding always on right side so it doesn't shift pts coordinates.
def pad_until_divisible(img,divisor):
    rs = array(img.shape)
    desired_rawsize = ceil(rs/divisor)*divisor
    padding = desired_rawsize - rs ## PAD ALL DIMS
    # padding = np.where(rawsize < patch_size, desired_rawsize - rawsize, 0) ## PAD ONLY SMALL DIMS
    img = np.pad(img, [(0,p) for p in padding], constant_values=0) ## WARN: must be the same const value that's used by torch model
    assert all(array(img.shape) % divisor == 0)
    return img, padding

## x : image::ndarray
## kind : in ['I','L'] for "Intensity" / "Label"
## colors : colormap
def img2png(x, kind, colors=None, normalize_intensity=True):
  
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
    cmap = array([(0,0,0)] + colors*256)[:256]
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

  ## combine XY YZ XZ projections into single image
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

  if kind=='L':
    x = _colorseg(x)
  else:
    if normalize_intensity:
      x = norm_minmax01(x)
    x = cmap(x)
  
  x = (x*255).astype(np.uint8)

  # white lines separating XY YZ XZ views
  if nd==3:
    x[b,:] = 255
    x[:,c] = 255

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

## Load `isbi-stats.csv` from disk and create a dictionary of params
## associated with a given `isbiname`
## WARNING: We attempt to interpret all cells in the table as python, and
## only fall back to `str` on failure. If we WANT a string, we have to wrap it in
## extra quotes, or ensure that `literal_eval()` fails.
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
      isbi = {k:parse(v) for k,v in row.items()}
      break

  isbi['voxelsize'] = array(isbi['voxelsize'])
  isbi['voxelsize'] = isbi['voxelsize'] / isbi['voxelsize'][-1]
  return isbi


### Core Functions

## Parameters for data(), train(), and predict()
def params(isbiname = "Fluo-C2DL-Huh7"):

  savedir = Path(f"cpnet-out/{isbiname}/")
  savedir.mkdir(parents=True,exist_ok=True)
  base = os.path.join(localinfo.local_base, isbiname)

  # np.random uses:
  # - initial assignment of train/vali labels
  # - shuffle train data every epoch
  np.random.seed(42)
  
  PR = SN()
  PR.isbiname = isbiname
  PR.savedir = savedir
  PR.ndim = 2 if "2D" in isbiname else 3

  isbi = load_isbi_csv(isbiname)
  PR.isbi = isbi
  
  if isbi['tname'] == 3:
    PR.name_raw = os.path.join(base, "{dset}/t{time:03d}.tif")
    PR.name_pts = os.path.join(base, "{dset}_GT/TRA/man_track{time:03d}.tif")
  elif isbi['tname'] == 4:
    PR.name_raw = os.path.join(base, "{dset}/t{time:04d}.tif")
    PR.name_pts = os.path.join(base, "{dset}_GT/TRA/man_track{time:04d}.tif")

  traindata = []
  testdata = []
  for d in ['01','02']:
    tb = isbi['times '+d]
    alltimes = range(tb[0], tb[1], isbi['subsample'])
    traindata += [dict(dset=d, time=t) for i,t in enumerate(alltimes) if i%8 in [4]]
    testdata  += [dict(dset=d, time=t) for i,t in enumerate(alltimes) if i%8 in [0]]
  
  PR.traindata = np.array(traindata)
  PR.testdata  = np.array(testdata)

  ## Are we evaluating or just predicting?
  PR.mode = 'withGT' # evaluate (requires GT)
  # PR.mode = 'NoGT' # just predict
  PR.run_tracking = True

  ## Colormap for "glances" during training.
  ## Predicted and GT points are single pixels of chosen color
  ## overlayed on raw image.
  cmap = np.zeros((256,3),np.float32)
  cmap[0] = (0,0,0) # background
  cmap[1] = (0,0,1) # prediction
  cmap[2] = (0,1,0) # ground truth
  cmap[3] = (1,0,0) # prediction + ground truth
  cmap = matplotlib.colors.ListedColormap(cmap)
  PR.cmap_glance = cmap  

  ## Define functions that are shared across at least two of:
  ## data(), train() or predict().

  # PR.zoom_img = lambda raw: zoom(raw, isbi['zoom'])

  if PR.ndim==2:

    PR.buildUNet = lambda : Unet3(16, [[1],[1]], pool=(2,2), kernsize=(5,5), finallayer=nn.Sequential)
    PR.divisor = (8,8)
    PR.splitIntoPatches = lambda x: splitIntoPatches(x, desired_outer_shape=(256,256), min_border_shape=(48,48), divisor=PR.divisor)
    PR.splitIntoPatchesPred = PR.splitIntoPatches
    # PR.zoom_img = lambda raw: zoom_img_make_divisible(raw, isbi['zoom'], divisor)

    ## train, predict
    PR.findPeaks = lambda x: peak_local_max(x, threshold_abs=.5, exclude_border=False, footprint=np.ones([5,5]))
    PR.snnMatch  = lambda yt, y: snnMatch(yt, y, dub=10, scale=[1,1]) ## y_true, y_predicted

  if PR.ndim==3:

    PR.buildUNet = lambda : Unet3(16, [[1],[1]], pool=(1,2,2), kernsize=(3,5,5), finallayer=nn.Sequential)
    PR.divisor = (1,8,8)
    # PR.splitIntoPatches = lambda x: splitIntoPatches(x, desired_outer_shape=(16,128,128), min_border_shape=(4,48,48), divisor=PR.divisor)
    PR.splitIntoPatches = lambda x: splitIntoPatches(x, desired_outer_shape=(32,256,256), min_border_shape=(4,16,16), divisor=PR.divisor)
    PR.splitIntoPatchesPred = lambda x: splitIntoPatches(x, desired_outer_shape=(32,400,400), min_border_shape=(8,48,48), divisor=PR.divisor)
    # PR.zoom_img = lambda raw: zoom_img_make_divisible(raw, isbi['zoom'], PR.divisor)
    
    ## train, predict
    PR.snnMatch  = lambda yt, y: snnMatch(yt, y, dub=100, scale=isbi['voxelsize'])
    PR.findPeaks = lambda x: peak_local_max(x, threshold_abs=.5, exclude_border=False, footprint=np.ones([3,5,5]))
  
  return PR

## construct training data
def data(PR):

  def f(dikt):
    raw = load_tif(PR.name_raw.format(**dikt)).astype(np.float32) ## cast from isbi data in u8 or u16
    lab = load_tif(PR.name_pts.format(**dikt))
    pts = array([x['centroid'] for x in regionprops(lab)])

    # raw, zoom2 = PR.zoom_img(raw)
    raw = zoom(raw,PR.isbi['zoom'])
    pts = zoom_pts(pts, PR.isbi['zoom'])

    ## cast f16 to reduce dataset size
    raw = norm_percentile01(raw,2,99.4).astype(np.float16)
    target = createTarget(pts, raw.shape, PR.isbi['sigma']).astype(np.float16)

    raw,padding = pad_until_divisible(raw,PR.divisor)
    target,padding = pad_until_divisible(target,PR.divisor)

    patches = PR.splitIntoPatches(raw.shape)
    raw_patches = [raw[p.outer] for p in patches]
    target_patches = [target[p.outer] for p in patches]
    samples = [SN(raw=r, target=t, inner=p.inner, outer=p.outer, inner_rel=p.inner_rel, **dikt) 
                  for r,t,p in zip(raw_patches,target_patches,patches)]

    # ipdb.set_trace()

    return samples

  data = [f(dikt) for dikt in PR.traindata]
  data = [s for dat in data for s in dat]

  if PR.isbi['sparse']:
    data = [s for s in data if s.target[s.inner_rel].max()>0.5]

  wipedir(PR.savedir/"data/")
  save_pkl(PR.savedir/"data/dataset.pkl", data)

  ## Save 10 pngs to give an overview of the training data.
  wipedir(PR.savedir/"data/png/")
  ids = ceil(np.linspace(0,len(data)-1,10)) if len(data)>10 else range(len(data)) ## <= 10 evenly sampled patches
  for i in ids:
    s = data[i]
    r = img2png(s.raw, 'I', normalize_intensity=True)
    mask = find_boundaries(s.target>0.5, mode='inner')
    t = img2png(mask.astype(np.uint8), 'L', colors=PR.cmap_glance) ## make borders blue
    composite = r.copy()
    m = np.any(t[:,:,:3]!=0 , axis=2)
    composite[m] = composite[m]/2.0 + t[m]/2.0 ## does not affect u8 dtype !
    save_png(PR.savedir/f'data/png/t{s.time:03d}-d{i:04d}.png', composite)

  # return data


## Train CPNET; Save history of validation metrics and best CPNET weights
## for each metric.
def train(PR, continue_training=False):

  CONTINUE = continue_training
  print("CONTINUE ? : ", bool(CONTINUE))

  print(f"""
    Begin training CP-Net on {PR.isbiname}
    Savedir is {PR.savedir / "train"}
    """)

  dataset = load_pkl(PR.savedir/"data/dataset.pkl")
  dataset = array(dataset)

  ## network, weights and optimization
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  print(f"We're using torch device {device} .")
  net = PR.buildUNet()
  net = net.to(device)
  init_weights(net)
  opt = torch.optim.Adam(net.parameters(), lr = 1e-4)

  ## Load weights and labels from disk, or create new labels.
  if CONTINUE:
    labels = load_pkl(PR.savedir / "train/labels.pkl")
    net.load_state_dict(torch.load(PR.savedir / f'train/m/best_weights_latest.pt', map_location=torch.device(device)))
    history = load_pkl(PR.savedir / 'train/history.pkl')
  else:
    wipedir(PR.savedir/'train/m')
    wipedir(PR.savedir/"train/glance_output_train/")
    wipedir(PR.savedir/"train/glance_output_vali/")
    N = len(dataset)
    a = (N*7)//8 ## split data into 7 parts training and 1 part validation
    labels = np.zeros(N,dtype=np.uint8)
    labels[a:] = 1 ## 0=train 1=vali
    np.random.shuffle(labels)
    save_pkl(PR.savedir / "train/labels.pkl", labels)
    history = SN(lossmeans=[], valimeans=[])

  # assert len(dataset)>8
  assert len(labels)==len(dataset)

  ## Add mask to each sample that removes the patches
  ## overlapping borders from the loss. For sparse data
  ## we set the mask to only include pixels within three
  ## kernel standard deviations of an annotated GT point.
  for s in dataset:
    s.weights = np.zeros(s.target.shape)
    s.weights[s.inner_rel] = 1
    if PR.isbi['sparse']:
      s.weights = (s.target > np.exp(-0.5*(3**2))).astype(np.float32) ## 3 std dev

  ## Define augmentation for training samples.
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

    ## TODO: random affine transform on raw intensity

    ## TODO: random jitter that leaves shape divisible by (1,8,8)?

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

    # ## Only apply loss on masked region.
    ### ERROR: THIS IS A BUG.
    # y  = y[s.inner_rel]
    # x  = x[s.inner_rel]
    # yt = yt[s.inner_rel]
    # w  = w[s.inner_rel]
    
    loss = torch.abs(((y-yt)**2)*w).mean()

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
    for i in idxs:
      s = validata[i]
      y, scores = mse_loss(s,augment=False,mode='vali')
      _valiscores.append((scores.loss, scores.f1, scores.height))

    history.valimeans.append(np.nanmean(_valiscores,axis=0))
    
    torch.save(net.state_dict(), PR.savedir / f'train/m/best_weights_latest.pt')

    valikeys   = ['loss','f1','height']
    valiinvert = [1,-1,-1] # minimize, maximize, maximize
    valis = array(history.valimeans).reshape([-1,3])*valiinvert

    for i,k in enumerate(valikeys):
      if np.nanmin(valis[:,i])==valis[-1,i]:
        torch.save(net.state_dict(), PR.savedir / f'train/m/best_weights_{k}.pt')

  def predGlances(epoch):
    ids = [0,N_train//2,N_train-1]
    for i in ids:
      composite = mse_loss(traindata[i], augment=True, mode='glance')
      save_png(PR.savedir/f'train/glance_output_train/a_{i:04d}_{min(epoch,10):03d}.png', composite)

    ids = [0,N_vali//2,N_vali-1]
    for i in ids:
      composite = mse_loss(validata[i], augment=False, mode='glance')
      save_png(PR.savedir/f'train/glance_output_vali/a_{i:04d}_{min(epoch,10):03d}.png', composite)


  ## Estimate the total time required for training.
  ## N_pix measures the Megapixels of `raw` data in traindata.
  N_pix = np.sum([np.prod(d.raw.shape) for d in traindata]) / 1_000_000 

  ## Rate has units of [megapixels / sec].
  ratemap = {
    (2,'cpu','Darwin') : 0.07 , 
    (3,'cpu','Darwin') : 0.032849 , 
    (2,'cpu','Linux')  : 0.42 , ## wow
    (3,'cpu','Linux')  : 0.154036 , 
    (2,'cuda','Linux') : 2.92 ,  ## wow
    (3,'cuda','Linux') : 0.976 , 
  }
  rate = ratemap.get( (PR.ndim, str(device), os.uname().sysname) , np.nan )
  print((PR.ndim, str(device), os.uname().sysname))

  N_completed = len(history.lossmeans)
  N_epochs = 100
  N_remaining = N_epochs - N_completed
  est_time = N_remaining*N_pix/60/60/rate
  print(f"Estimated Time: {N_remaining} epochs * {N_pix:.2f} Mpix / {rate:.2f} Mpix/s = {est_time:.2f}h \n")
  print(f"\nBegin training until epoch {N_epochs}...\n\n")

  for ep in range(N_completed, N_epochs):
    tic = time()
    trainOneEpoch()
    validateOneEpoch()
    save_pkl(PR.savedir / "train/history.pkl", history)
    if ep in range(0,10,2) or ep%10==0: predGlances(ep)
    dt  = time() - tic

    print("\033[F",end='') ## move cursor UP one line 
    print(f"finished epoch {ep+1}/{N_epochs}, loss={history.lossmeans[-1]:4f}, dt={dt:4f}, rate={N_pix/dt:5f} Mpix/s", end='\n',flush=True)


## Evaluate models and make predictions on unseen data.
def predict(PR):

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  net = PR.buildUNet()
  net = net.to(device)

  def predsingle(dikt):
    raw = load_tif(PR.name_raw.format(**dikt)) #.transpose([1,0,2,3])[1]
    # raw, zoom2 = PR.zoom_img(raw)
    raw = zoom(raw,PR.isbi['zoom'])
    raw = norm_percentile01(raw,2,99.4)
    raw, padding = pad_until_divisible(raw,PR.divisor)

    ## Seamless prediction tiling
    pred = np.zeros(raw.shape)
    for p in PR.splitIntoPatchesPred(pred.shape):
      x = torch.Tensor(raw[p.outer][None,None]).to(device)
      with torch.no_grad():
        pred[p.inner] = net(x).cpu().numpy()[0,0][p.inner_rel]
    ss = tuple([slice(0,s-p) for s,p in zip(pred.shape, padding)])
    pred = pred[ss]

    ## Find peaks and transform them back to original image size.
    height = pred.max()    
    pts = PR.findPeaks(pred)
    pts = zoom_pts(pts , 1 / array(PR.isbi['zoom']))
    
    if PR.mode=='withGT':
      lab = load_tif(PR.name_pts.format(**dikt))
      gtpts = np.array([x['centroid'] for x in regionprops(lab)])
      matching = PR.snnMatch(gtpts, pts)

    def makePNG():
      r = img2png(raw, 'I')
      m = pred>0.5
      p = img2png(labelComponents(m)[0], 'L') #, colors=plt.cm.magma)
      composite = r.copy()
      composite[m] = (r[m]/2 + p[m]/2).astype(np.uint8).clip(min=0,max=255)
      return composite
    # composite = makePNG()

    return SN(**locals())

  # wipedir(PR.savedir / "predict")
  wipedir(PR.savedir / "predict/scores")
  wipedir(PR.savedir / "predict/pred")
  N_imgs = len(PR.testdata)
  ltps = []
  matching_results = []
  badkeys = ['gt_matched_mask', 'yp_matched_mask', 'gt2yp', 'yp2gt', 'pts_gt', 'pts_yp']

  ## Make predictions for each saved weight set : 'latest','loss','f1','height'.
  ## This allows for simple model ensembling.
  # weight_list = ['latest','loss','f1','height']
  weight_list = ['f1']
  for weights in weight_list:
    net.load_state_dict(torch.load(PR.savedir / f'train/m/best_weights_{weights}.pt', map_location=torch.device(device)))

    for i, dikt in enumerate(PR.testdata):
      print(f"Predicting on image {i+1}/{N_imgs}...", end='\r',flush=True)
      d = predsingle(dikt)
      ltps.append(d.pts)
      save_png(PR.savedir/"predict/pred/t-{dset}-{time:04d}-{weights}.png".format(**dikt,weights=weights), img2png(d.pred, 'I', colors=plt.cm.magma))
      
      if PR.mode=='withGT':
        print(dedent(f"""
          weights : {weights}
             time : {dikt['time']:d}
               f1 : {d.matching.f1:.3f}
        precision : {d.matching.precision:.3f}
           recall : {d.matching.recall:.3f}
        """))
        res = {**dikt, 'weights':weights, **{k:v for k,v in d.matching.__dict__.items() if k not in badkeys}}
        matching_results.append(res)

  if PR.mode=='withGT':
    save_pkl(PR.savedir/"predict/scores/matching.pkl", matching_results)

  if PR.run_tracking == False: sys.exit(0)

  print(f"Run tracking...", end='\n', flush=True)
  tb = tracking2.nn_tracking(ltps, aniso=PR.isbi['voxelsize'])

  ## Draw a graph of the cell lineage tree with nodes colored
  ## according to the ISBI standard.
  if False:
    tracking2.draw(tb)
    plt.ion()
    plt.show()
    input()

  ## random colormap for tracking
  cmap_track = np.random.rand(256,3).clip(min=0.2)
  cmap_track[0] = (0,0,0)
  cmap_track = matplotlib.colors.ListedColormap(cmap_track)

  wipedir(PR.savedir/"track/png")
  for i, dikt in enumerate(PR.testdata):
    print(f"Saving image {i+1}/{N_imgs}...", end='\r',flush=True)
    raw = load_tif(PR.name_raw.format(**dikt)).astype(np.float32)
    rawpng = img2png(raw, 'I')
    ## WARNING: Using index `i` for time instead of dikt['time'].
    ## This allows us to track across arbitrary sequences of images
    ## to easily test the robustness of the tracker.
    lab = tracking2.createTarget(tb, i, raw.shape, PR.isbi['sigma']) 
    labpng = img2png(lab, 'L', colors=cmap_track)
    composite = np.round(rawpng/2 + labpng/2).astype(np.uint8).clip(min=0,max=255)
    save_png(PR.savedir/"track/png/img{time:03d}.png".format(**dikt), composite)


if __name__=="__main__":

  isbiname = sys.argv[1]

  PR = params(isbiname)

  DTP = sys.argv[2]

  if 'D' in DTP:
    data(PR)
  if 'Tc' in DTP:
    train(PR, continue_training=1)
  elif 'T' in DTP:
    train(PR, continue_training=0)
  if 'P' in DTP:
    predict(PR)





