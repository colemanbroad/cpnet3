## standard lib
from pathlib import Path
from time import time
from types import SimpleNamespace as SN
import json
import pickle
import shutil
from glob import glob

## standard scipy
import ipdb
import matplotlib
import numpy as np
import torch

from matplotlib                import pyplot as plt
from scipy.ndimage             import label as labelComponents
from scipy.ndimage             import zoom
from skimage.feature           import peak_local_max
from skimage.io                import imsave
from skimage.measure           import regionprops
from skimage.morphology.binary import binary_dilation
from tifffile                  import imread

## 3rd party 
import augmend
from pykdtree.kdtree import KDTree as pyKDTree
from textwrap import dedent

## segtools
from segtools.point_matcher import match_unambiguous_nearestNeib as snnMatch
from segtools import torch_models
from segtools.cpnet_utils import createTarget, splitIntoPatches

"""
RUN ME ON SLURM!!
sbatch -J e23-mau -p gpu --gres gpu:1 -n 1 -t  6:00:00 -c 1 --mem 128000 -o slurm_out/e23-mau.out -e slurm_err/e23-mau.out --wrap '/bin/time -v python e23_mauricio2.py'
"""

# savedir = Path("/Users/broaddus/Desktop/mpi-remote/project-broaddus/devseg_2/expr/e23_mauricio/v02/")
# savedir = Path("/Users/broaddus/Desktop/work/bioimg-collab/mau-2021/data-experiment/")
# savedir = Path("/projects/project-broaddus/devseg_2/expr/e23_mauricio/v03/")
savedir = Path("data/")

def load_tif(name): return imread(name) 
def load_pkl(name): 
  with open(name,'rb') as file:
    return pickle.load(file)
def save_pkl(name, stuff):
  with open(name,'wb') as file:
    pickle.dump(stuff, file)
def save_json(name, stuff):
  with open(name,'w') as file:
    json.dump(stuff, file)
def save_png(name, img):
    imsave(name, img)


def plotHistory():

  history = load_pkl(savedir/"train/history.pkl")
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

"""Parameters for data(), train(), and predict()"""
def params():
  D = SN()

  ## data, predict
  D.name_raw = "Fluo-C2DL-Huh7/01/t{time:03d}.tif"
  D.name_pts = "Fluo-C2DL-Huh7/01_GT/TRA/man_track{time:03d}.tif"
  D.zoom = (0.5,0.5)

  ## train, predict
  D.nms_footprint = [5,5]

  ## data
  D.outer_shape_train = (128,128)
  D.min_border_shape_train = (16,16)
  D.sigma = (5,5)
  D.traintimes = [0] #range(25)

  ## train
  D.border = [0,0]
  D.match_dub = 10
  D.match_scale = [1,1]
  D.ndim = 2

  ## predict
  D.mode = 'NoGT' ## 'withGT'
  D.predtimes = [29] #range(29,30)

  ## ------------------------------------------------

  ## data-01/
  # D.outer_shape_train = (128,128)
  # D.min_border_shape_train = (16,16)

  D.outer_shape_train = (64,64)
  D.min_border_shape_train = (0,0)

  return D


"""
Tiling patches with overlapping borders. Requires loss masking.
Const outer size % 8 = 0.
"""
def data_v03():

  D = params()

  def f(i):
    raw = load_tif(D.name_raw.format(time=i)) #.transpose([1,0,2,3])
    lab = load_tif(D.name_pts.format(time=i)) #.transpose([])
    pts = np.array([x['centroid'] for x in regionprops(lab)])
    raw = zoom(raw, D.zoom, order=1)
    # lab = zoom(lab, D.zoom, order=1)
    pts = zoom_pts(pts, D.zoom)
    raw = norm_percentile01(raw,2,99.4)
    target = createTarget(pts, raw.shape, D.sigma)
    patches = splitIntoPatches(raw.shape, outer_shape=D.outer_shape_train, min_border_shape=D.min_border_shape_train)
    raw_patches = [raw[p.outer] for p in patches]
    target_patches = [target[p.outer] for p in patches]
    samples = [SN(raw=r, target=t, inner=p.inner, outer=p.outer, inner_rel=p.inner_rel, time=i) 
                  for r,t,p in zip(raw_patches,target_patches,patches)]
    return samples

  # return pickle.load(open(str(savedir / 'data/filtered.pkl'), 'rb'))

  data = [f(i) for i in D.traintimes]
  data = [s for dat in data for s in dat]

  save_pkl("dataset.pkl", data)

  ## save train/vali/test data
  wipedir(savedir/"data/png/")
  for i,s in enumerate(data[::10]):
    r = img2png(s.raw)
    t = img2png(s.target, colors=plt.cm.magma)
    composite = r//2 + t//2 
    imsave(savedir/f'data/png/t{s.time:03d}-d{i:04d}.png', composite)

  return data

"""
NOTE: train() includes additional data filtering.
"""
def train(dataset=None,continue_training=False):

  CONTINUE = continue_training
  print("CONTINUE ? : ", bool(CONTINUE))

  print(f"""
    Begin training CP-Net on Fluo-C2DL-Huh7/01
    Savedir is {savedir / "train"}
    """)

  dataset = np.array(dataset)

  ## NOTE these params only required for validation!
  D = params()

  ## network, weights and optimization
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  net = torch_models.Unet3(16, [[1],[1]], pool=(2,2),   kernsize=(5,5),   finallayer=torch_models.nn.Sequential)
  net = net.to(device)
  torch_models.init_weights(net)
  opt = torch.optim.Adam(net.parameters(), lr = 1e-4)

  # ipdb.set_trace()

  # if str(device)=='cpu':
  #   dataset = dataset[::23]

  ## load weights and sample train/vali assignment from disk?
  if CONTINUE:
    labels = load_pkl(savedir / "train/labels.pkl")
    net.load_state_dict(torch.load(savedir / f'train/m/best_weights_latest.pt'))
    history = load_pkl(savedir / 'train/history.pkl')
  else:
    wipedir(savedir/'train/m')
    wipedir(savedir/"train/glance_output_train/")
    wipedir(savedir/"train/glance_output_vali/")
    N = len(dataset)
    # a, b = (N*5)//8, (N*7)//8  ## MYPARAM train / vali / test 
    a = (N*7)//8 ## don't use test patches. test on full images.
    labels = np.zeros(N,dtype=np.uint8)
    labels[a:]=1; ## labels[b:]=2 ## 0=train 1=vali 2=test
    np.random.shuffle(labels)
    save_pkl(savedir / "train/labels.pkl", labels)
    history = SN(lossmeans=[], valimeans=[])

  assert len(dataset)>8
  assert len(labels)==len(dataset)

  ## augmentation by flips and rotations
  def build_augmend(ndim):
    ag = augmend
    aug = ag.Augmend()
    aug.add([ag.FlipRot90(axis=0), ag.FlipRot90(axis=0), ag.FlipRot90(axis=0),], probability=1)
    aug.add([ag.FlipRot90(axis=(1,2)), ag.FlipRot90(axis=(1,2)), ag.FlipRot90(axis=(1,2))], probability=1)
    # aug.add([IntensityScaleShift(), Identity(), Identity()], probability=0.5)
    # aug.add([AdditiveNoise(), Identity(), Identity()], probability=0.5)
    # ## continuous rotations that introduce black regions
    # ## this will make our weights non-binary, but that's OK.
    # aug.add([Rotate(axis=(1,2), order=1), Rotate(axis=(1,2), order=1), Rotate(axis=(1,2), order=1)], probability=1)
    # aug.add([Elastic(axis=(1,2), order=1), Elastic(axis=(1,2), order=1), Elastic(axis=(1,2), order=1)], probability=1)
    return aug
  f_aug = build_augmend(D.ndim)

  ## add loss weights to each patch
  for s in dataset:
    s.weights = np.ones(s.target.shape)
    # s.weights = binary_dilation(s.target>0 , np.ones((1,7,7)))
    # s.weights = (s.target > 0)
    # print("{:.3f}".format(s.weights.mean()),end="  ")

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
    if augment:
      x,yt,w = f_aug([x,yt,w])

    # ## glance at patches after augmentation
    # r = img2png(x)
    # p = img2png(yt,colors=plt.cm.magma)
    # # t = img2png(w,colors=plt.cm.magma)
    # composite = np.round(r/2 + p/2).astype(np.uint8).clip(min=0,max=255)
    # # m = np.any(t[:,:,:3]!=0 , axis=2)
    # # composite[m] = t[m]
    # save(composite,savedir/f'train/glance_augmented/a{s.time:03d}_{i:03d}.png')

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
      pts    = peak_local_max(y, threshold_abs=.5, exclude_border=False, footprint=np.ones(D.nms_footprint))
      gt_pts = peak_local_max(yt.detach().cpu().numpy().astype(np.float32), threshold_abs=.5, exclude_border=False, footprint=np.ones(D.nms_footprint))
      
      ## filter border points
      patch_shape   = np.array(x.shape)
      pts2    = [p for p in pts if np.all(p%(patch_shape-D.border) > D.border)]
      gt_pts2 = [p for p in gt_pts if np.all(p%(patch_shape-D.border) > D.border)]

      matching = snnMatch(gt_pts2, pts2, dub=D.match_dub, scale=D.match_scale)
      scores = SN(loss=loss, f1=matching.f1, height=y.max())
      return y, scores

    if mode=='glance':
      r = img2png(x.numpy())
      p = img2png(y.numpy(),colors=plt.cm.magma)
      t = img2png((yt > 0.9).numpy().astype(np.uint8))
      composite = np.round(r/2 + p/2).astype(np.uint8).clip(min=0,max=255)
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
      y,loss = mse_loss(s, augment=False, mode='train')
      _losses.append(loss)
      dt = time()-tic; tic = time()
      print(f"it {i}/{N_train}, dt {dt:5f}, max {y.max():5f}", end='\r',flush=True)

    history.lossmeans.append(np.nanmean(_losses))

  def validateOneEpoch():
    _valiscores = []
    idxs = np.arange(N_vali)
    np.random.shuffle(idxs)
    # idxs = idxs[:len(idxs)//10]
    for i in idxs:
      s = validata[i] ## no idxs
      y, scores = mse_loss(s,augment=False,mode='vali')
      _valiscores.append((scores.loss, scores.f1, scores.height))
      # if i%10==0: print(f"_scores",_scores, end='\n',flush=True)

    history.valimeans.append(np.nanmean(_valiscores,axis=0))

    torch.save(net.state_dict(), savedir / f'train/m/best_weights_latest.pt')

    valikeys   = ['loss','f1','height']
    valiinvert = [1,-1,-1] # minimize, maximize, maximize
    valis = np.array(history.valimeans).reshape([-1,3])*valiinvert

    for i,k in enumerate(valikeys):
      if np.nanmin(valis[:,i])==valis[-1,i]:
        torch.save(net.state_dict(), savedir / f'train/m/best_weights_{k}.pt')

  def predGlances(time):
    ids = [0,N_train//2,N_train-1]
    for i in ids:
      s = traindata[i]
      composite = mse_loss(s, augment=False, mode='glance')
      save_png(savedir/f'train/glance_output_train/a{time:03d}_{i:03d}.png', composite)

    ids = [0,N_vali//2,N_vali-1]
    for i in ids:
      s = validata[i]
      composite = mse_loss(s, augment=False, mode='glance')
      save_png(savedir/f'train/glance_output_vali/a{time:03d}_{i:03d}.png', composite)

  n_pix = np.sum([np.prod(d.raw.shape) for d in traindata]) / 1_000_000 ## Megapixels of raw data in traindata
  rate = 1 if str(device)!='cpu' else 0.0435 ## megapixels / sec 
  N_epochs=300 ## MYPARAM
  print(f"Estimated Time: {n_pix} Mpix * 1s/Mpix = {300*n_pix/60/rate:.2f}m = {300*n_pix/60/60/rate:.2f}h \n")
  print(f"\nBegin training for {N_epochs} epochs...\n\n")

  for ep in range(N_epochs):
    tic = time()
    trainOneEpoch()
    validateOneEpoch()
    save_pkl(savedir / "train/history.pkl", history)
    if ep in range(10) or ep%10==0: predGlances(ep)
    dt  = time() - tic

    print("\033[F",end='') ## move cursor UP one line 
    print(f"finished epoch {ep+1}/{N_epochs}, loss={history.lossmeans[-1]:4f}, dt={dt:4f}, rate={n_pix/dt:5f} Mpix/s", end='\n',flush=True)


"""
Make predictions for each saved weight set : 'latest','loss','f1','height'
Include avg/min across predictions too! Simple model ensembling.
"""
def predict():

  wipedir(savedir / "predict")

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  net = torch_models.Unet3(16, [[1],[1]], pool=(2,2),   kernsize=(5,5),   finallayer=torch_models.nn.Sequential)
  net = net.to(device)

  D = params()

  def predsingle(time):
    raw = load_tif(D.name_raw.format(time=i)) #.transpose([1,0,2,3])[1]
    raw = zoom(raw, D.zoom,order=1)
    raw = norm_percentile01(raw,2,99.4)

    if D.mode=='withGT':
      lab = load_tif(D.name_pts.format(time=i)) #.transpose([])
      pts = np.array([x['centroid'] for x in regionprops(lab)])
      gtpts = [p for i,p in enumerate(gtpts) if classes[i] in ['p','pm']]
      gtpts = (np.array(gtpts) * D.zoom).astype(np.int)

    pred = np.zeros(raw.shape)
    for p in splitIntoPatches(pred.shape, (512,512), (24,24)):
      x = torch.Tensor(raw[None,None][p.outer]).to(device)
      with torch.no_grad():
        pred[p.inner] = net(x).cpu().numpy()[0,0][p.inner_rel]

    height = pred.max()
    
    pts = peak_local_max(pred,threshold_abs=.2,exclude_border=False,footprint=np.ones(D.nms_footprint))
    pts = zoom_pts(pts , 1 / np.array(D.zoom))
    
    if D.mode=='withGT':
      matching = snnMatch(gtpts,pts,dub=100,scale=[3,1,1])
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

  for weights in ['f1']: #['latest','loss','f1','height']:
    net.load_state_dict(torch.load(savedir / f'train/m/best_weights_{weights}.pt'))

    for i in D.predtimes:
      d = predsingle(i)
      save_png(savedir/f"predict/t{i:04d}-{weights}.png", d.composite)

if __name__=="__main__":
  dataset = data_v03()
  train(dataset,continue_training=1)
  predict()