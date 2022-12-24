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

## standard scipy
import ipdb
import matplotlib
import numpy as np
from numpy import array, exp, zeros, maximum, indices
def ceil(x): return np.ceil(x).astype(int)
def floor(x): return np.floor(x).astype(int)
import torch

from matplotlib                import pyplot as plt
from scipy.ndimage             import label as labelComponents
from scipy.ndimage             import zoom
from skimage.feature           import peak_local_max
from skimage.io                import imsave
from skimage.measure           import regionprops
from skimage.morphology.binary import binary_dilation
from tifffile                  import imread

from isbidata import isbi_times, isbi_scales

## 3rd party 
# import augmend

## segtools
from torch_models import Unet3, init_weights, nn
from match import snnMatch

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
  imsave(name, img)

def plotHistory():

  D = params()

  # history = load_pkl("/Users/broaddus/Desktop/mpi-remote/project-broaddus/cpnet3/main/Fluo-N3DH-CHO/train/history.pkl")
  history = load_pkl(D.savedir/"train/history.pkl")
  fig, ax = plt.subplots(nrows=4,sharex=True, )

  ax[0].plot(np.log(history.lossmeans), label="log train loss")
  ax[0].legend()

  valis = np.array(history.valimeans)

  ax[0+1].plot(np.log(valis[:,0]), label="log vali loss")
  ax[0+1].legend()

  ax[1+1].plot(valis[:,1], label="f1")
  ax[1+1].legend()

  ax[2+1].plot(valis[:,2], label="height")
  ax[2+1].legend()

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
  outer shape % 8 == 0 in XY (last two dims)
  img_shape   >= outer_shape 
  outer_shape >= 2*min_border_shape

  GUARANTEES
  outer shape == outer_shape forall patches
  inner shape <= outer_shape - min_border_shape
  acutal boder shape >= min_border_shape forall patches except on image bounds
  max(inner shape) - min(inner shape) <= 1 forall shapes
  
  """
  
  img_shape = array(img_shape)
  outer_shape = array(outer_shape)
  min_border_shape = array(min_border_shape)
  divisor = array(divisor)
  
  # make shapes divisible by 8
  assert all(outer_shape % divisor == 0), f"Error: `outer_shape` {outer_shape}%8 != 0."
  assert all(img_shape>=outer_shape), f"Error: `outer_shape` doesn't fit"
  assert all(outer_shape>=2*min_border_shape), f"Error: borders too wide"

  # our actual shape will be <= this desired shape. `outer_shape` is fixed,
  # but the border shape will grow.
  desired_inner_shape = outer_shape - min_border_shape

  ## round up, shrinking the actual inner_shape
  inner_counts = ceil(img_shape / desired_inner_shape)
  inner_shape_float = img_shape / inner_counts

  def f(i,n):
    a = inner_shape_float[n]
    b = floor(a*i)      # slice start
    c = floor(a*(i+1))  # slice stop
    inner = slice(b,c)
    
    w = c-b

    # shift `outer` patch inwards when we hit a border to maintain outer_shape.
    if b==0:                          # left border
      outer = slice(0,outer_shape[n])
      inner_rel = slice(0,w)
    elif c==img_shape[n]:             # right border 
      outer = slice(img_shape[n]-outer_shape[n],img_shape[n])
      inner_rel = slice(outer_shape[n]-w,outer_shape[n])
    else:
      r = b - min_border_shape[n]
      outer = slice(r,r+outer_shape[n])
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

def zoom_pts(pts,scale):
  """
  rescale pts to be consistent with scipy.ndimage.zoom(img,scale)
  """
  # assert type(pts) is np.ndarray
  pts = pts+0.5                         ## move origin from middle of first bin to left boundary (array index convention)
  pts = pts * scale                     ## rescale
  pts = pts-0.5                         ## move origin back to middle of first bin
  pts = np.round(pts).astype(np.uint32) ## binning
  return pts

def img2png(x,colors=None):

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
  D = x.ndim

  if D==3:
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
    x = norm_minmax01(x)
    x = cmap(x)
  
  x = (x*255).astype(np.uint8)

  if D==3:
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



"""
Parameters for data(), train(), and predict()
"""

def params(
  isbiname = "Fluo-C2DL-Huh7"
  ):
  D = SN()

  savedir = Path(f"main/{isbiname}/")

  # savedir = Path("/Users/broaddus/Desktop/mpi-remote/project-broaddus/devseg_2/expr/e23_mauricio/v02/")
  # savedir = Path("/Users/broaddus/Desktop/work/bioimg-collab/mau-2021/data-experiment/")
  # savedir = Path("/projects/project-broaddus/devseg_2/expr/e23_mauricio/v03/")

  D.isbiname = isbiname
  D.savedir = savedir

  D.ndim = 2 if "2D" in isbiname else 3
  ## data, predict
  
  # base = f"/projects/project-broaddus/rawdata/isbi_train/{isbiname}/"
  base = f"{isbiname}/"
  tb = isbi_times[isbiname]
  D.name_raw = base + "{dset}/t{time:03d}.tif"
  D.name_pts = base + "{dset}_GT/TRA/man_track{time:03d}.tif"
  
  ## data
  # D.traintimes = [dict(dset=d, time=t) for t in range(0,91,17) for d in ["01","02"]]
  ## predict
  # D.predtimes =  [dict(dset=d, time=t) for t in range(3,91,17) for d in ["01","02"]]

  alldata = np.array([dict(dset=d, time=t) 
                        for d in ['01','02']
                        for t in range(tb[d][0], tb[d][1])])

  np.random.seed(42)
  np.random.shuffle(alldata)
  alldata = alldata[:16]
  N = len(alldata)
  D.traintimes = alldata[:N*7//8]
  # D.predtimes = alldata[N*7//8:]
  D.predtimes = np.array([dict(dset=d, time=t) 
                        for d in ['01']
                        for t in range(tb[d][0], tb[d][0] + 4)])
  
  ## predict
  D.mode = 'NoGT' ## 'withGT'

  if D.ndim==2:
    ## data, predict
    D.zoom = (0.25, 0.25)
    ## train, predict
    D.findPeaks = lambda x: peak_local_max(x, threshold_abs=.5, exclude_border=False, footprint=np.ones([5,5]))
    D.snnMatch  = lambda yt, y: snnMatch(yt, y, dub=10, scale=[1,1])
    D.buildUNet = lambda : Unet3(16, [[1],[1]], pool=(2,2), kernsize=(5,5), finallayer=nn.Sequential)
    ## data
    # print("shape is : ", )
    D.splitIntoPatches = lambda x: splitIntoPatches(x, outer_shape=(128,128), min_border_shape=(16,16), divisor=(8,8))
    D.sigma = (5,5)
    ## train
    D.border = [0,0]

  if D.ndim==3:
    ## data, predict
    D.zoom = (1, 0.25, 0.25)
    ## train, predict
    aniso = np.array(isbi_scales[isbiname])
    aniso = aniso / aniso[2] # 
    D.snnMatch  = lambda yt, y: snnMatch(yt, y, dub=10, scale=aniso)
    D.buildUNet = lambda : Unet3(16, [[1],[1]], pool=(1,2,2), kernsize=(3,5,5), finallayer=nn.Sequential)
    D.findPeaks = lambda x: peak_local_max(x, threshold_abs=.5, exclude_border=False, footprint=np.ones([3,5,5]))
    ## data
    D.splitIntoPatches = lambda x: splitIntoPatches(x, outer_shape=(16,128,128), min_border_shape=(4,16,16), divisor=(1,8,8))
    D.sigma = (3,5,5)
    ## train
    D.border = [0,0,0]

  if "Fluo-N3DH-CHO" in D.name_raw: # shape = (5,111,128)
    D.splitIntoPatches = lambda x: splitIntoPatches(x, outer_shape=(5, 64, 128), min_border_shape=(0,16,0), divisor=(1,8,8))

  D.splitIntoPatchesPred = D.splitIntoPatches

  ## ------------------------------------------------

  ## main-01/
  # D.outer_shape_train = (128,128)
  # D.min_border_shape_train = (16,16)

  ## main-02/
  # D.outer_shape_train = (64,64)
  # D.min_border_shape_train = (0,0)

  return D


"""
Tiling patches with overlapping borders. Requires loss masking.
Const outer size % 8 = 0.
"""
def data(D):

  def f(dikt):
    raw = load_tif(D.name_raw.format(**dikt)) #.transpose([1,0,2,3])
    lab = load_tif(D.name_pts.format(**dikt)) #.transpose([])
    pts = np.array([x['centroid'] for x in regionprops(lab)])
    raw = zoom(raw, D.zoom, order=1)
    # lab = zoom(lab, D.zoom, order=1)
    pts = zoom_pts(pts, D.zoom)
    raw = norm_percentile01(raw,2,99.4)
    target = createTarget(pts, raw.shape, D.sigma)
    print("rawshape: ", raw.shape)
    patches = D.splitIntoPatches(raw.shape)
    raw_patches = [raw[p.outer] for p in patches]
    target_patches = [target[p.outer] for p in patches]
    samples = [SN(raw=r, target=t, inner=p.inner, outer=p.outer, inner_rel=p.inner_rel, **dikt) 
                  for r,t,p in zip(raw_patches,target_patches,patches)]
    return samples

  # return pickle.load(open(str(D.savedir / 'data/filtered.pkl'), 'rb'))

  data = [f(dikt) for dikt in D.traintimes]
  data = [s for dat in data for s in dat]

  wipedir(D.savedir/"data/")
  save_pkl(D.savedir/"data/dataset.pkl", data)

  ## save train/vali/test data
  wipedir(D.savedir/"data/png/")
  for i,s in enumerate(data[::10]):
    r = img2png(s.raw)
    t = img2png(s.target, colors=plt.cm.magma)
    composite = r//2 + t//2 
    imsave(D.savedir/f'data/png/t{s.time:03d}-d{i:04d}.png', composite)

  return data

"""
NOTE: train() includes additional data filtering.
"""
def train(D, dataset=None,continue_training=False):

  CONTINUE = continue_training
  print("CONTINUE ? : ", bool(CONTINUE))

  print(f"""
    Begin training CP-Net on {D.isbiname}
    Savedir is {D.savedir / "train"}
    """)

  dataset = np.array(dataset)

  ## network, weights and optimization
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  net = D.buildUNet()
  net = net.to(device)
  init_weights(net)
  opt = torch.optim.Adam(net.parameters(), lr = 1e-4)

  ## load weights and sample train/vali assignment from disk?
  if CONTINUE:
    labels = load_pkl(D.savedir / "train/labels.pkl")
    net.load_state_dict(torch.load(D.savedir / f'train/m/best_weights_latest.pt'))
    history = load_pkl(D.savedir / 'train/history.pkl')
  else:
    wipedir(D.savedir/'train/m')
    wipedir(D.savedir/"train/glance_output_train/")
    wipedir(D.savedir/"train/glance_output_vali/")
    N = len(dataset)
    # a, b = (N*5)//8, (N*7)//8  ## MYPARAM train / vali / test 
    a = (N*7)//8 ## don't use test patches. test on full images.
    labels = np.zeros(N,dtype=np.uint8)
    labels[a:]=1; ## labels[b:]=2 ## 0=train 1=vali 2=test
    np.random.shuffle(labels)
    save_pkl(D.savedir / "train/labels.pkl", labels)
    history = SN(lossmeans=[], valimeans=[])

  assert len(dataset)>8
  assert len(labels)==len(dataset)

  ## add loss weights to each patch
  for s in dataset:
    s.weights = np.zeros(s.target.shape)
    s.weights[s.inner_rel] = 1

    # s.weights = binary_dilation(s.target>0 , np.ones((1,7,7)))
    # s.weights = (s.target > 0)
    # print("{:.3f}".format(s.weights.mean()),end="  ")

  ## augmentation by flips and rotations
  def augmentSample(sample):
    s = sample
    
    # rotate -90 + flip x
    if np.random.rand() < 0.5:
      for x in [s.raw, s.target, s.weights]:
        x = x.transpose([1,0]) if D.ndim==2 else x.transpose([0,2,1])
    
    # flip y[z]
    if np.random.rand() < 0.5:
      for x in [s.raw, s.target, s.weights]:
        x = x[::-1]
    
    # flip x[y]
    if np.random.rand() < 0.5:
      for x in [s.raw, s.target, s.weights]:
        x = x[:, ::-1]

    # flip z
    if np.random.rand() < 0.5 and D.ndim==3:
      for x in [s.raw, s.target, s.weights]:
        x = x[:, :, ::-1]


  # if D.sparse:
  #   # w0 = dgen.weights__decaying_bg_multiplier(s.target,0,thresh=np.exp(-0.5*(3)**2),decayTime=None,bg_weight_multiplier=0.0)
  #   # NOTE: i think this is equivalent to a simple threshold mask @ 3xstddev, i.e.
  #   w0 = (s.target > np.exp(-0.5*(3**2))).astype(np.float32)
  # else:
  #   w0 = np.ones(s.target.shape,dtype=np.float32)
  # return w0
  # df['weights'] = df.apply(addweights,axis=1)

  ## Filter dataset
  # traindata = df[(df.labels==0) & (df.npts>0)] ## MYPARAM subsample traindata ?
  # tmax = np.array([s.tmax for s in D.samples])
  traindata = dataset[labels==0] # & (tmax > 0.99)]
  validata  = dataset[labels==1] # & (tmax > 0.99)]
  # if s.tmax < 0.99 and np.random.rand()<0.99: continue

  N_train = len(traindata)
  N_vali = len(validata)
  print(f"""
    Data filtered from N={len(dataset)} to 
    N_train={N_train} , N_vali={N_vali}
    """)


  def mse_loss(s, augment=True, mode=None):
    assert mode in ['train', 'vali', 'glance']

    x  = s.raw.copy()
    yt = s.target.copy()
    w  = s.weights.copy()

    ## remove the border regions that make our patches a bad size
    if False:
      divis = (1,8,8)
      ss = [[None,None,None],[None,None,None],[None,None,None],]
      for n in [0,1,2]:
        rem = x.shape[n]%divis[n]
        if rem != 0:
          ss[n][0] = 0
          ss[n][1] = -rem
      ss = tuple([slice(a,b,c) for a,b,c in ss])

    ## makes patches divisible by (1,8,8) (simpler than above)
    if False:
      ps = np.floor(np.array(x.shape)/(1,8,8)) * (1,8,8)
      ss = tuple([slice(0,n) for n in ps])
      for arr in [x,yt,w]: arr = arr[ss] # does this work?

    ## augmentation
    if augment: augmentSample(s)

    # ## glance at patches after augmentation
    # r = img2png(x)
    # p = img2png(yt,colors=plt.cm.magma)
    # # t = img2png(w,colors=plt.cm.magma)
    # composite = np.round(r/2 + p/2).astype(np.uint8).clip(min=0,max=255)
    # # m = np.any(t[:,:,:3]!=0 , axis=2)
    # # composite[m] = t[m]
    # save(composite,D.savedir/f'train/glance_augmented/a{s.time:03d}_{i:03d}.png')

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
      pts = D.findPeaks(y)
      gt_pts = D.findPeaks(yt.detach().cpu().numpy().astype(np.float32))
      
      ## filter border points
      patch_shape   = np.array(x.shape)
      pts2    = [p for p in pts if np.all(p%(patch_shape-D.border) > D.border)]
      gt_pts2 = [p for p in gt_pts if np.all(p%(patch_shape-D.border) > D.border)]

      matching = D.snnMatch(gt_pts2, pts2)
      scores = SN(loss=loss, f1=matching.f1, height=y.max())
      return y, scores

    if mode=='glance':
      r = img2png(x.cpu().numpy())
      p = img2png(y.cpu().numpy(),colors=plt.cm.magma)
      w = img2png(w.cpu().numpy())
      t = img2png((yt > 0.9).cpu().numpy().astype(np.uint8))
      composite = np.round(r/3 + p/3 + w/3).astype(np.uint8).clip(min=0,max=255)
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

    torch.save(net.state_dict(), D.savedir / f'train/m/best_weights_latest.pt')

    valikeys   = ['loss','f1','height']
    valiinvert = [1,-1,-1] # minimize, maximize, maximize
    valis = np.array(history.valimeans).reshape([-1,3])*valiinvert

    for i,k in enumerate(valikeys):
      if np.nanmin(valis[:,i])==valis[-1,i]:
        torch.save(net.state_dict(), D.savedir / f'train/m/best_weights_{k}.pt')

  def predGlances(time):
    ids = [0,N_train//2,N_train-1]
    for i in ids:
      composite = mse_loss(traindata[i], augment=True, mode='glance')
      save_png(D.savedir/f'train/glance_output_train/a{time:03d}_{i:03d}.png', composite)

    ids = [0,N_vali//2,N_vali-1]
    for i in ids:
      composite = mse_loss(validata[i], augment=False, mode='glance')
      save_png(D.savedir/f'train/glance_output_vali/a{time:03d}_{i:03d}.png', composite)

  n_pix = np.sum([np.prod(d.raw.shape) for d in traindata]) / 1_000_000 ## Megapixels of raw data in traindata
  if D.ndim==2:
    rate = 1.4 if str(device)!='cpu' else 0.0435 ## megapixels / sec 
  elif D.ndim==3:
    rate = 1.0 if str(device)!='cpu' else 0.0310 ## megapixels / sec 
  N_epochs=300 ## MYPARAM
  print(f"Estimated Time: {n_pix} Mpix / {rate} Mpix/s = {n_pix/rate/60*N_epochs:.2f}m = {300*n_pix/60/60/rate:.2f}h \n")
  print(f"\nBegin training for {N_epochs} epochs...\n\n")

  for ep in range(N_epochs):
    tic = time()
    trainOneEpoch()
    validateOneEpoch()
    save_pkl(D.savedir / "train/history.pkl", history)
    if ep in range(10) or ep%10==0: predGlances(ep)
    dt  = time() - tic

    print("\033[F",end='') ## move cursor UP one line 
    print(f"finished epoch {ep+1}/{N_epochs}, loss={history.lossmeans[-1]:4f}, dt={dt:4f}, rate={n_pix/dt:5f} Mpix/s", end='\n',flush=True)

"""
Make predictions for each saved weight set : 'latest','loss','f1','height'
Include avg/min across predictions too! Simple model ensembling.
"""
def predict(D):

  wipedir(D.savedir / "predict")

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  net = D.buildUNet()
  net = net.to(device)

  def predsingle(dikt):
    raw = load_tif(D.name_raw.format(**dikt)) #.transpose([1,0,2,3])[1]
    rawshape = raw.shape
    raw = zoom(raw, D.zoom,order=1)
    raw = norm_percentile01(raw,2,99.4)

    if D.mode=='withGT':
      lab = load_tif(D.name_pts.format(**dikt)) #.transpose([])
      pts = np.array([x['centroid'] for x in regionprops(lab)])
      gtpts = [p for i,p in enumerate(gtpts) if classes[i] in ['p','pm']]
      gtpts = (np.array(gtpts) * D.zoom).astype(np.int)

    ## seamless prediction tiling
    pred = np.zeros(raw.shape)
    for p in D.splitIntoPatchesPred(pred.shape):
      x = torch.Tensor(raw[p.outer][None,None]).to(device)
      with torch.no_grad():
        pred[p.inner] = net(x).cpu().numpy()[0,0][p.inner_rel]

    ## find and scale peaks back to orig space
    height = pred.max()    
    pts = D.findPeaks(pred)
    pts = zoom_pts(pts , 1 / np.array(D.zoom))

    # ## filter border points
    # pred_shape   = np.array(pred.shape)
    # pts2    = [p for p in pts if np.all(p%(pred_shape-D.border) > D.border)]
    # gt_pts2 = [p for p in gt_pts if np.all(p%(pred_shape-D.border) > D.border)]
    
    if D.mode=='withGT':
      matching = D.snnMatch(gtpts,pts)
      print(dedent(f"""
          weights : {weights}
             time : {time:03d}
               f1 : {matching.f1:.3f}
        precision : {matching.precision:.3f}
           recall : {matching.recall:.3f}
        """))

    def f():
      r = img2png(raw)
      p = img2png(labelComponents(pred>0.5)[0]) #, colors=plt.cm.magma)
      composite = np.round(r/2 + p/2).astype(np.uint8).clip(min=0,max=255)
      # ipdb.set_trace()
      # t = img2png((yt > 0.9).numpy().astype(np.uint8))
      # m = np.any(t[:,:,:3]!=0 , axis=2)
      # composite[m] = t[m]
      return composite
    composite = f()

    return SN(**locals())

  ltps = []
  rawshape = None
  for weights in ['latest']: #['latest','loss','f1','height']:
    net.load_state_dict(torch.load(D.savedir / f'train/m/best_weights_{weights}.pt'))

    for dikt in D.predtimes:
      d = predsingle(dikt)
      ltps.append(d.pts)
      save_png(D.savedir/"predict/t{time:04d}-{weights}.png".format(**dikt,weights=weights), d.composite)
      rawshape = d.rawshape

  import tracking
  wipedir(D.savedir/"track")
  track_labeled_images = tracking.makeISBILabels(ltps,rawshape)

  cmap = np.random.rand(256,3).clip(min=0.2)
  cmap[0] = (0,0,0)
  cmap = matplotlib.colors.ListedColormap(cmap)

  # for time, lab in enumerate():
  for i, dikt in enumerate(D.predtimes):
    raw = img2png(load_tif(D.name_raw.format(**dikt)).astype(np.float32))
    lab = img2png(track_labeled_images[i], colors=cmap)
    save_png(D.savedir/"track/z{time:03d}.png".format(**dikt), track_labeled_images[i])
    composite = np.round(raw/2 + lab/2).astype(np.uint8).clip(min=0,max=255)
    save_png(D.savedir/"track/c{time:03d}.png".format(**dikt), composite)
    save_png(D.savedir/"track/r{time:03d}.png".format(**dikt), raw)
    save_png(D.savedir/"track/l{time:03d}.png".format(**dikt), lab)




if __name__=="__main__":
  arg = sys.argv[1]
  if arg:
    D = params(arg)
  else:
    D = params()
  dataset = data(D)
  train(D, dataset, continue_training=0)
  predict(D)

