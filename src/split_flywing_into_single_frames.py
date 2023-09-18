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
# from skimage.io                import imsave
from skimage.measure           import regionprops
from skimage.morphology.binary import binary_dilation
from skimage.segmentation      import find_boundaries
from tifffile                  import imread,imsave


os.chdir("/Users/broaddus/work/isbi/cpnet3/")
raw = imread("data-noisbi/care_flywing_crops/gt_initial_stack.tif")
gt_segment = imread("data-noisbi/care_flywing_crops/gt_true_track_consistent.tif")



for i, _ in enumerate(raw):
	gtframe = gt_segment[i]
	mask = np.any(gtframe[:,:,:3]!=255 , axis=2)
	lab = labelComponents(mask)[0]
	# imsave(f"data-noisbi/care_flywing_crops/raw/raw{i:03d}.tif", raw[i])
	imsave(f"data-noisbi/care_flywing_crops/gt/gt{i:03d}.tif", lab)


