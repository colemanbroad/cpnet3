# /Users/broaddus/Desktop/work/cpnet3/data-raw/Fluo-C3DH-A549/01_GT/SEG/man_seg027.tif

import numpy as np
import re
from types import SimpleNamespace as SN
import json
from glob import glob
from skimage.measure  import regionprops
from pathlib import Path
import isbidata

# from isbi_tools import get_isbi_info, isbi_datasets
# from experiments_common import parse_pid
# from segtools.ns2dir import load,save

from tifffile import imread
def load_tif(name): return imread(name) 

import pickle
def load_pkl(name): 
    with open(name,'rb') as file:
        return pickle.load(file)

base = "/projects/project-broaddus/rawdata/isbi-train/"
base = "data-raw/"
base2 = "/Users/broaddus/Desktop/mpi-remote/project-broaddus/devseg_2/data/seginfo/"

def analyze_single(labname):
    _time,_zpos = re.search(r"man_seg_?(\d{3,4})_?(\d{3,4})?\.tif",labname).groups()
    _time = int(_time)
    if _zpos: _zpos = int(_zpos)
    lab = load_tif(labname)
    rp = regionprops(lab)
    rp = SN(bbox=np.array([x['bbox'] for x in rp]),centroid=np.array([x['centroid'] for x in rp]))
    D = lab.ndim
    rp.boxdims = rp.bbox[:,D:] - rp.bbox[:,:D]
    return SN(lab=labname,time=_time,zpos=_zpos,rp=rp)

def run():
  for isbiname in isbidata.isbi_by_size:
    for dataset in ['01','02']:
        labnames = sorted(glob(base + f"{isbiname}/{dataset}_GT/SEG/*.tif"))
        for labname in labnames:
            seginfo = analyze_single(labname)

            # name = base2 + f"seginfo-{isbiname}-{dataset}.pkl"
            # try:
            #     dat = load_pkl(name)
            #     print(dat.rp.boxdims)
            # except:
            #     print(f"Missing {name}")

def run2():
    files = glob("/Users/broaddus/Desktop/mpi-remote/project-broaddus/devseg_2/data/seginfo/*.pkl")
    for name in files:
        print(name)
        dat = load_pkl(name)
        print( np.mean([d.rp.boxdims.mean(0) for d in dat], axis=0) )

        # for x in [d.rp.boxdims.mean(0) for d in dat]:
        #     print(x)

if __name__=='__main__':
    run2()

