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

# img = load_tif("/Users/broaddus/Desktop/work/cpnet3/data-isbi/Fluo-C3DH-A549/01/t000.tif").astype(np.float32)

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

## The pooling is strange and nonuniform. The division into blocks is as even as possible, 
## And the remainder is distributed as evenly as possible using overlaps of adjacent blocks
## Every 
x = torch.Tensor(np.arange(5))
y = torch.nn.functional.adaptive_max_pool1d(x[None,None], 2)


# x = np.arange(20):
# print(20/11)
# print(zoom())

# img = load_tif("/Users/broaddus/Desktop/work/cpnet3/data-isbi/DIC-C2DH-HeLa/01/t000.tif")

# res = norm_percentile01(zoom(img.astype(np.float32),0.25,order=1,prefilter=True) , 2, 99.4)
# save_tif(savedir / "convert-norm-zoom0.25-order1-pftrue.tif", res)
# res = norm_percentile01(zoom(img.astype(np.float32),0.25,order=1,prefilter=False) , 2, 99.4)
# save_tif(savedir / "convert-norm-zoom0.25-order1-pffalse.tif",res)

# for imgname in glob("/Users/broaddus/Desktop/work/cpnet3/data-isbi/*/01/t000.tif"):
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








