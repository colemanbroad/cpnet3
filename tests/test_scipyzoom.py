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



from glob import glob

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

def norm_percentile01(x,p0,p1):
  lo,hi = np.percentile(x,[p0,p1])
  if hi==lo: 
    return x-hi
  else: 
    return (x-lo)/(hi-lo)



import skimage
from cpnet import img2png
import torch


savedir = Path("test_scipyzoom")
# wipedir(savedir)

# img = load_tif("/Users/broaddus/Desktop/work/cpnet3/data-raw/Fluo-C3DH-A549/01/t000.tif").astype(np.float32)

# res = skimage.measure.block_reduce(img, (1,2,2), np.mean)
# save_tif(savedir / f"a549-br-2.png", img2png(res, 'I', normalize_intensity=True))
# res = skimage.measure.block_reduce(img, (1,3,3), np.mean)
# save_tif(savedir / f"a549-br-3.png", img2png(res, 'I', normalize_intensity=True))
# res = skimage.measure.block_reduce(img, (1,4,4), np.mean)
# save_tif(savedir / f"a549-br-4.png", img2png(res, 'I', normalize_intensity=True))

# res = skimage.measure.block_reduce(img, (1,4,4), np.mean)
# save_tif(savedir / f"a549-br-skimage.png", img2png(res, 'I', normalize_intensity=True))

# res = torch.nn.functional.avg_pool3d(torch.Tensor(img)[None], (1,4,4))[0].numpy()
# save_tif(savedir / f"a549-br-torch.png", img2png(res, 'I', normalize_intensity=True))

# for i,x in enumerate(np.linspace(0.25, 0.33, 10)):
#   res = zoom(img,(1,x,x),order=1,prefilter=True)
#   save_tif(savedir / f"a549-{i}.png", img2png(res, 'I', normalize_intensity=True))
#   # res = norm_percentile01(res , 2, 99.4)
#   # save_tif(savedir / f"a549-{i}-2.png", img2png(res, 'I', normalize_intensity=True))
#   # res = zoom(img,(1,0.28,0.28),order=1,prefilter=False)
#   # save_tif(savedir / f"a549-{i}-3.png", img2png(res, 'I', normalize_intensity=True))
#   # res = norm_percentile01(res , 2, 99.4)
#   # save_tif(savedir / f"a549-{i}-4.png", img2png(res, 'I', normalize_intensity=True))


"""
There are so many ways to upscale and downscale images.
What kinds of scaling are goverened by sampling theory?
What makes a function "scaling"? Is there a definition?

As we zoom "in" (upscaling) there may be arbitrarily many choices we make for filling in the missing data.
Computational super resolution is a class of techniques. There are more choices than n-order spline interpolation.

For zoom "out" (downscaling) we are more constrained. 
The eye takes some kind of an average over the field of view, so zoom out should be some kind of average.

But we can perform various kinds of local averaging directly, or by prefiltering + sampling, or by a full 
conversion to Fourier space, cutting off or extrapolating high frequencies, and then back transforming.

scipy.ndimage.zoom() ## with/out prefilter
torch.nn.functional.avg_pool()

- How do these mathematical concepts relate to the functions at our disposal?
- How do these functions affect the "domain" of the image?
- Can we demonstrate aliasing problems with good test examples?




"""


## The adaptive_max_pool1d is strange and nonuniform. The division into blocks is as even as possible, 
## And the remainder is distributed as evenly as possible using overlaps of adjacent blocks.

N = 10
M = 15
x0 = torch.Tensor(np.arange(N))
y0 = torch.rand(N)
plt.plot(x0,y0,'-o')

x1 = np.linspace(0,N-1,M)
y1 = torch.nn.functional.adaptive_avg_pool1d(y0[None,None], M)[0,0]
plt.plot(x1,y1,'-o')


# x = np.arange(20):
# print(20/11)
# print(zoom())

# img = load_tif("/Users/broaddus/Desktop/work/cpnet3/data-raw/DIC-C2DH-HeLa/01/t000.tif")

# res = norm_percentile01(zoom(img.astype(np.float32),0.25,order=1,prefilter=True) , 2, 99.4)
# save_tif(savedir / "convert-norm-zoom0.25-order1-pftrue.tif", res)
# res = norm_percentile01(zoom(img.astype(np.float32),0.25,order=1,prefilter=False) , 2, 99.4)
# save_tif(savedir / "convert-norm-zoom0.25-order1-pffalse.tif",res)

# for imgname in glob("/Users/broaddus/Desktop/work/cpnet3/data-raw/*/01/t000.tif"):
# 	img = load_tif(imgname)
# 	print(img.dtype)
# assert False

# res = norm_percentile01(zoom(img,0.25,order=1,prefilter=True) , 2, 99.4)
# save_tif(savedir / "normzoom0.25-order1-pftrue.tif", res)
# res = norm_percentile01(zoom(img,0.25,order=1,prefilter=False) , 2, 99.4)
# save_tif(savedir / "normzoom0.25-order1-pffalse.tif",res)

# save_tif(savedir / "zoom0.50-order1-pftrue.tif", zoom(img,0.50,order=1,prefilter=True))
# save_tif(savedir / "zoom0.50-order1-pffalse.tif",zoom(img,0.50,order=1,prefilter=False))

# save_tif(savedir / "zoom0.25-order1.tif",        zoom(img,0.25,order=1))
# save_tif(savedir / "zoom0.25-order1-pftrue.tif", zoom(img,0.25,order=1,prefilter=True))
# save_tif(savedir / "zoom0.25-order1-pffalse.tif",zoom(img,0.25,order=1,prefilter=False))
# save_tif(savedir / "zoom0.25-order3.tif",        zoom(img,0.25,order=3))
# save_tif(savedir / "zoom0.25-order3-pftrue.tif", zoom(img,0.25,order=3,prefilter=True))
# save_tif(savedir / "zoom0.25-order3-pffalse.tif",zoom(img,0.25,order=3,prefilter=False))








