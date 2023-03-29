import numpy as np
import ipdb
from types import SimpleNamespace as SN
from pykdtree.kdtree import KDTree
import matplotlib
from pointmatch import snnMatch, build_scores
from cpnet import load_isbi_csv
import os
import pickle

from tifffile                  import imread
import re
from skimage.measure           import regionprops
from os import path
from tabulate import tabulate

from glob import glob

from numpy import array, exp, zeros, maximum, indices
def ceil(x): return np.ceil(x).astype(int)
def floor(x): return np.floor(x).astype(int)

## walk down the nodes starting at root and moving on to children
def addIsbiLabels(tb):

  children = dict()
  for n,p in tb.parents.items():
    if p is None: continue
    children[p] = children.get(p, []) + [n]
  
  ## ISBI Labels for each node by incrementing a global track_id
  global_track_id = 1
  ## dict that maps from existing node ID to Isbi ID
  toisbi = dict()

  ## Find all tree roots
  nodestack = [n for n,p in tb.parents.items() if p is None]

  ## label each node in the tree; when cells divide, inc the global_track_id;
  ## preorder traversal is 1. root, 2. left subtree, 3. right subtree
  while len(nodestack)>0:
    global_track_id += 1
    s = nodestack.pop()

    while True:
      toisbi[s] = global_track_id
      cs = children.get(s, None)
      if cs is None: break ## ended with cell death or movie stop

      if len(cs)==1:
        ## track continues
        s = cs[0]
      elif len(cs)>1:
        ## end with division
        nodestack += cs
        break
  
  tb.children = children
  tb.toisbi = toisbi

## ltps: list of pts for each time
## aniso: the pixel/voxel size (relative)
## dub: distance upper bound for child-parent connections (in pixels)
## When a node (time,label) has no parent we map it to (time-1, 0)
## i.e. label zero serves as the background label
## We use the index into ltps as the initial label, but convert
## this label to the ISBI scheme for most processing
def nn_tracking(*,dtps,aniso=(1,1),dub=100):
  parents = dict()
  pts_dict = dict()

  times = sorted(list(dtps.keys()))

  ## put all the points in a big dictionary
  for t,pts in dtps.items(): 
    for i,p in enumerate(pts):
      pts_dict[(t,i)] = p

  t0 = times[0]
  for idx, _ in enumerate(dtps[t0]): parents[(t0,idx)] = None


  # ipdb.set_trace()

  ## for each pts(t) connect all the points to nearest neib
  ## in the previous pts(t-1).
  for t_idx in range(1,len(times)):
    t = times[t_idx]
    pts = dtps[t]
    t_prev = times[t_idx-1]
    ## WARN: do we want t_idx-1 or t-1 ? t_idx-1 connects across many frames.
    pts_prev = dtps[t_prev]

    kdt = KDTree(array(pts_prev)*aniso)
    dist, curr2prev = kdt.query(array(pts)*aniso, k=1, distance_upper_bound=dub)
    for idx_curr,idx_prev in enumerate(curr2prev):
      p = None if idx_prev==len(pts_prev) else (t_prev,idx_prev)
      parents[(t,idx_curr)] = p

  tb = SN(parents=parents, pts=pts_dict)
  conformTracking(tb)
  return tb

## Variant of cpnet.createTarget()
## tb: TrueBranching
## time: int
## img_shape: tuple
## sigmas: tuple of radii for label marker
def createTargetWithTrackingLabels(tb, time, img_shape, sigmas):
  s  = array(sigmas).astype(float)
  ks = floor(7*s).astype(int)   ## extend support to 7/2 sigma in every direc
  ks = ks - ks%2 + 1            ## enfore ODD shape so kernel is centered! 

  # pts = array(pts).astype(int)
  ## FIXME
  nodes = tb.time2labelset[time]

  # lab = array([l for n,l in tb.toisbi.items() if n[0]==time])
  # pts = array([p for n,p in tb.pts.items() if n[0]==time])
  pts = array([tb.pts[(time,l)] for l in nodes])
  lab = [tb.toisbi[(time,l)] for l in nodes]

  if len(pts)==0: return zeros(img_shape).astype(np.int64)

  ## create a single kernel patch
  def f(x):
    x = x - (ks-1)/2
    return (x*x/s/s).sum() <= 1 ## radius squared
    # return exp(-(x*x/s/s).sum()/2)
  kern = array([f(x) for x in indices(ks).reshape((len(ks),-1)).T]).reshape(ks)
    
  target = zeros(ks + img_shape).astype(np.int64) ## include border padding
  w = ks//2                      ## center coordinate of kernel
  pts_offset = pts + w           ## offset by padding

  for i,p in enumerate(pts_offset):
    target_slice = tuple(slice(a,b+1) for a,b in zip(p-w,p+w))
    target[target_slice] = maximum(target[target_slice], kern*lab[i]) ## `maximum` puts labels on top of background (zero)

  remove_pad = tuple(slice(a,a+b) for a,b in zip(w,img_shape))
  target = target[remove_pad]

  return target

## draw tails on centerpoints to show cell motion
def createTailsWithTrackingLabels(tb, time, img_shape):
  imgbase = zeros(img_shape).astype(np.int64)
  if time==0: return imgbase

  if len(tb.time2labelset[time])==0: return imgbase

  for l0 in tb.time2labelset[time]:
    p0 = tb.pts[(time,l0)]
    n1 = tb.parents[(time,l0)]
    if n1 is None: continue
    p1 = tb.pts[n1]
    color = 255
    drawLine(imgbase,p0,p1,color)

  return imgbase

## Classic Bresenham Algorithm
## Draw a line from pt -> par 
def drawLine(img,pt,par,color):
  x0,y0 = int(pt[0]), int(pt[1])
  x1,y1 = int(par[0]), int(par[1])
  dx = abs(x1 - x0)
  dy = abs(y1 - y0)
  x, y = x0, y0
  sx = -1 if x0 > x1 else 1
  sy = -1 if y0 > y1 else 1
  if dx > dy:
    err = dx / 2.0
    while x != x1:
      img[x,y] = color
      err -= dy
      if err < 0:
        y += sy
        err += dx
      x += sx
  else:
    err = dy / 2.0
    while y != y1:
      img[x,y] = color
      err -= dx
      if err < 0:
        x += sx
        err += dy
      y += sy
  img[x,y] = color

## directory: path of e.g. 01_GT/TRA/
def loadISBITrackingFromDisk(directory):
  ## WARN: must cast to int explicitly, because type(np.uint + int) is numpy.float64 !
  lbep = np.loadtxt(directory + '/man_track.txt').astype(int)
  if lbep.ndim == 1: lbep = lbep.reshape([1,4])

  ## Filter to remove mistakes in man_track.txt
  lbep_filtered = []
  for l,b,e,p in lbep:
    if not b<=e:
      print(f"ISBI Error. l={l},b={b},e={e},p={p}")
      continue
    lbep_filtered.append((l,b,e,p))
  lbep = lbep_filtered

  mantrackdict = {m[0]:SN(b=m[1],e=m[2],p=m[3]) for m in lbep}

  pts = dict()
  for i, imgname in enumerate(glob(directory + "/man_track*.tif")):
    time = int(re.search(r'man_track(\d+)\.tif',imgname).group(1))
    lab  = imread(imgname)
    pts  = {**pts, **{(time, x['label']): x['centroid'] for x in regionprops(lab)}}

  parent = dict()
  for l,b,e,p in lbep:

    ## add the node to TB the first time it appears
    ## WARNING: Can't simply use b-1 for parent time! There can be gaps in tracks.
    parent[(b,l)] = (mantrackdict[p].e , p) if p!=0 else None #(b-1,p)
    # if (b-1,p) in {(70, 56), (76, 41)}: ipdb.set_trace()

    ## for all subsequent times we know the "parent" node exists
    for time in range(b+1, e+1):
      parent[(time,l)] = (time-1,l)

  tb = SN(parents=parent, pts=pts)
  conformTracking(tb)
  return tb

## draw the nodes with compressed spatial + time axis
def drawLineageTree(tb):
  # find clusters of nodes that are a part of the same family tree
  # they should be drawn together
  # nodestack = [n for n,p in tb.parents.items() if p is None]
  # clusters = []

  ## nodes sorted by position on x axis
  nodes = [x[0] for x in sorted([[k,*v] for k,v in tb.pts.items() if tb.parents[k] is None], key=lambda x: x[-1])]

  cmap = np.random.rand(256,3).clip(min=0.2)
  cmap[0] = (0,0,0)
  cmap = matplotlib.colors.ListedColormap(cmap)

  ## map labels to colors
  def l2c(label):
    if label==0:
      return cmap(0)
    else:
      return cmap(label % 254 + 1)

  plt.figure()
  import matplotlib.collections as mc
  linecollection = mc.LineCollection([], colors='k', linewidths=2)
  # fig, ax = pl.subplots()
  plt.gca().add_collection(linecollection)

  lines = []

  # starting x positions are equally spaced on the x axis
  xpos = {n:3*i for i,n in enumerate(nodes)}

  # as we move through the child nodes we plot them with y positions 
  # given by time and x positions equal to their parents + some random factor
  while len(nodes) > 0:

    times = [n[0] for n in nodes]
    xs = [xpos[n] for n in nodes]
    color = [l2c(tb.toisbi[n]) for n in nodes]

    plt.scatter(xs,times,c=color)

    # replace nodes with children. if multiple children, then offset them
    # slightly from parent x position with a random value
    _nodes = [] 
    for i,n in enumerate(nodes): 
      cs = tb.children.get(n, [])
      _nodes += cs
      if len(cs)==1:
        m = cs[0]
        p = tb.parents[m]
        x_p = xpos[p]
        xpos[m] = x_p
        lines.append(array([(x_p, p[0]), (xpos[m], m[0])]))
      else:
        for i,m in enumerate(cs):
          p = tb.parents[m]
          x_p = xpos[p] 
          xpos[m] = x_p + [-0.2, 0.3, 0.4][i]
          lines.append(array([(x_p, p[0]), (xpos[m], m[0])]))
    nodes = _nodes
    linecollection.set_segments(lines)

  plt.gca().invert_yaxis()
  plt.show()

import matplotlib.pyplot as plt
from random import random

## Add time2labelset and assert that all the nodes
## have an associated spatial point.
def conformTracking(tb):
  d = dict()
  for time,label in tb.pts.keys():
    d[time] = d.get(time, set()) | {label}
  tb.time2labelset = d
  tb.times = sorted(list(tb.time2labelset.keys()))

  assert tb.pts.keys() == tb.parents.keys()
  pk = tb.parents.keys()
  # pv = {v for v in tb.parents.values() if v[0]!=-1 and v[1]!=0}
  pv = {v for v in tb.parents.values() if v is not None}
  assert pv-pk==set()

def evalNNTrackingOnIsbiGTDirectory(directory):
  isbiname, dataset = path.normpath(directory).split(path.sep)[-3:-1]
  isbi = load_isbi_csv(isbiname)
  # directory = "data-isbi/DIC-C2DH-HeLa/01_GT/TRA/"
  gt = loadISBITrackingFromDisk(directory)
  dtps = {t:[gt.pts[(t,n)] for n in gt.time2labelset[t]] for t in gt.times}
  # aniso = [1,1] if '2D' in directory else [1,1,1]
  aniso = isbi['voxelsize']
  dub = 100
  yp = nn_tracking(dtps=dtps, aniso=aniso, dub=dub)
  # ipdb.set_trace()
  scores = compare_trackings(gt,yp,aniso,dub)
  return {(isbiname, dataset) : scores}

def evalAllDirs():
  res = dict()
  # for dire in glob("../data-isbi/*/*_GT/TRA/"):

  base = path.normpath("../cpnet-out/")
  for directory in sorted(glob("/projects/project-broaddus/rawdata/isbi_train/*/*_GT/TRA/")):
    print(directory)
    isbiname, dataset = path.normpath(directory).split(path.sep)[-3:-1]
    outdir = path.join(base, isbiname, dataset, 'track-analysis')
    os.makedirs(outdir, exist_ok=True)
    # ipdb.set_trace()

    try:
      assert False
      d = pickle.load(open(outdir + '/compare_results-aniso.pkl','rb'))
      if len(d)==0:
        print("Empty Data!")
        assert False
      # pickle.dump(d, open(outdir + '/compare_results-aniso.pkl','wb'))
      # os.remove(directory + '/compare_results-aniso.pkl')
    except:
      d = evalNNTrackingOnIsbiGTDirectory(directory)
      pickle.dump(d, open(outdir + '/compare_results-aniso.pkl','wb'))

    res.update(d)
  return res

def formatres(res):
  lines = []
  for k,v in res.items():
    d = dict(name=k[0], dset=k[1][:2])
    d.update({'node-'+k2:v2 for k2,v2 in v.node.__dict__.items()})
    d.update({'edge-'+k2:v2 for k2,v2 in v.edge.__dict__.items()})
    lines.append(d)

  print(tabulate(lines, headers='keys'))
  return lines

## Adds `time2labelset` to gt and yp.
## Returns global set similarity scores for nodes and edges
## across the entire timeseries
def compare_trackings(gt,yp,aniso,dub):

  times = sorted(list(gt.time2labelset.keys() | yp.time2labelset.keys()))
  
  ## maybe find point matches should work with maps instead of arrays...
  ## the default should be that we pass around unique id's for pts... i.e. label, (time,label), (gt,time,label), etc
  ## an index into a dense array must always be mapped to/from. is always arbitrary. but sometimes all we have!
  def f(t):
    _gt = {(t,l):gt.pts[(t,l)] for l in gt.time2labelset[t]}
    _yp = {(t,l):yp.pts[(t,l)] for l in yp.time2labelset[t]}
    # ipdb.set_trace()
    return snnMatch(_gt, _yp, dub=dub, scale=aniso)
  node_matches = {t:f(t) for t in times}

  ## TODO: do more with the scores for all timepoints

  node_totals = build_scores(
    n_m  = sum([x.n_matched for x in node_matches.values()]),
    n_p  = sum([x.n_proposed for x in node_matches.values()]),
    n_gt = sum([x.n_gt for x in node_matches.values()]),
    )

  ## count edges. if parent label == 0 => cell appearance
  n_edges_gt = len([v for v in gt.parents.values() if v != None])
  n_edges_yp = len([v for v in yp.parents.values() if v != None])

  ## Now we find matching edges by going pt -> parent -> match == pt -> match -> parent (it's a commutative diagram)
  edge_matches = dict()
  # gt_edges_nomatch = set()
  for t in times[1:]:
    for n1,n2 in node_matches[t].matches.items():
      p1 = gt.parents[n1]
      p2 = yp.parents[n2]
      # ipdb.set_trace()
      if p1 is None or p2 is None:
        continue

      ## WARN: use p1[0] instead of i-1 because of gaps in tracks!
      if node_matches[p1[0]].matches.get(p1,None)==p2:
        edge_matches[frozenset({p1,n1})] = frozenset({p2,n2})

      # else:
      #   print(f"No matching parents {n1}->{p1} != {n2}->{p2}")
      #   # ipdb.set_trace()
      #   gt_edges_nomatch |= {frozenset({p1,n1})}

  edge_totals = build_scores(n_m=len(edge_matches), n_p=n_edges_yp, n_gt=n_edges_gt)

  return SN(node=node_totals, edge=edge_totals)



