import numpy as np
import ipdb
from types import SimpleNamespace as SN
from pykdtree.kdtree import KDTree
import matplotlib
from pointmatch import snnMatch, build_scores
from cpnet import load_isbi_csv
import os
import pickle

import matplotlib.collections as mc

import matplotlib.pyplot as plt
from random import random

from tifffile                  import imread
import re
from skimage.measure           import regionprops
from os import path
from tabulate import tabulate

from glob import glob

from numpy import array, exp, zeros, maximum, indices
def ceil(x): return np.ceil(x).astype(int)
def floor(x): return np.floor(x).astype(int)


# Add the ISBI labeling `tb.toisbi` to the tracking. Create the labels by
# forming a stack of unlabeled nodes and walking down the lineage tree depth
# first.
def addIsbiLabels2(tb):
  
  # ISBI Labels for each node by incrementing a global track_id
  currentmax = 1
  # dict that maps from existing node ID to Isbi ID
  labels = dict()

  # Stack of unprocessed nodes. Initialize with all lineage tree roots.
  nodestack = [n for n,p in tb.parents.items() if p is None]

  # Label each node in the tree. When we reach a cell division add both
  # daughters to the nodestack and pop next node from stack.
  while len(nodestack)>0:
    currentmax += 1
    s = nodestack.pop()

    while True:
      labels[s] = currentmax
      cs = tb.children.get(s, None)
      if cs is None: break # ended with cell death or movie stop

      if len(cs)==1:
        # track continues
        s = cs[0]
      elif len(cs)>1:
        # end with division
        nodestack += cs
        break
  
  tb.toisbi = labels

# Add the ISBI labeling `tb.toisbi` to the tracking via a nested loop over
# times and nodes[time]. The inverse `isbi2nodeset` makes it easy to find
# short stubs in the lineage tree via 
# 
#           sorted(tb.isbi2nodeset.values(), key=lambda v: len(v))
# 
def addIsbiLabels(tb):
  isbi2nodeset = dict()
  labels = dict()
  currentmax = 1
  for t in tb.time2labelset.keys():
    for idx in tb.time2labelset[t]:

      node = (t,idx)
      # if node has no parent or does have siblings: assign new ID. 
      parent = tb.parents[node]
      if parent is None or len(tb.children[parent])>1:
        labels[node] = currentmax
        isbi2nodeset[currentmax] = {node}
        currentmax += 1
        continue

      # otherwise inherit ID from parent
      labels[node] = labels[parent]
      isbi2nodeset[labels[parent]] |= {node}

  tb.toisbi = labels
  tb.isbi2nodeset = isbi2nodeset

# Classify all nodes into 'enter', 'exit', and 'move' and classify all links
# into 'move' and 'divide'.
def addNodeClassification(tb):

  label2nodeset = dict(enter=set(), exit=set(), move=set())
  labels = dict()
  edge_label2nodeset = dict(move=set(), divide=set())
  edge_labels = dict()

  for t in tb.time2labelset.keys():
    for idx in tb.time2labelset[t]:

      node = (t,idx)
      parent = tb.parents[node]
      children = tb.children.get(node,None)

      if parent is None:
        # if node has no parent it's an enter
        labels[node] = 'enter'
        label2nodeset['enter'] |= {node}
      elif children is None:
        # if node has no children it's an exit
        labels[node] = 'exit'
        label2nodeset['exit'] |= {node}
      else:
        # otherwise it's a move
        labels[node] = 'move'
        label2nodeset['move'] |= {node}

      if parent is None: continue

      if len(tb.children[parent])==1:
        # if node has no siblings it's link to parent is a movement
        edge_labels[node] = 'move'
        edge_label2nodeset['move'] |= {node}
      else:
        # otherwise it's got siblings and it's link to parent is a division
        edge_labels[node] = 'divide'
        edge_label2nodeset['divide'] |= {node}

  tb.label2nodeset = label2nodeset
  tb.labels = labels
  tb.edge_label2nodeset = edge_label2nodeset
  tb.edge_labels = edge_labels

def remNodeClassification(tb):
  del tb.label2nodeset
  del tb.labels
  del tb.edge_label2nodeset
  del tb.edge_labels

# Remove singleton branches which are usually false divisions.
def pruneSingletonBranches(tb):

  # Singleton branches have no children and either:
  #  no parent. The node appears and then dies.
  #  yes parent. The node is likely a false division resulting from FP detection.

  for (k,ns) in tb.isbi2nodeset.items():
    if len(ns) != 1: continue
    node = list(ns)[0]
    parent = tb.parents[node]
    children = tb.children.get(node,None)

    if children is not None:
      # This is the one (rare) case where we DONT delete the node
      continue

    ## we can just do this
    del tb.pts[node]
    del tb.parents[node]
    tb.children[parent].remove(node) ## in-place
    
    continue

    ## if we want to try and fix the isbi labeling in place instead of 
    ## regenerating it we can do this... Otherwise regen everything.
    if len(tb.children[parent]) != 1: continue
    
    onlychild = list(tb.children[parent])[0]
    child_label = tb.toisbi[onlychild]
    parent_label = tb.toisbi[parent]
    for n in tb.isbi2nodeset[child_label]:
      tb.toisbi[n] = parent_label

  ## now regenerate state

  # del tb.edge_label2nodeset
  # del tb.edge_labels
  # del tb.label2nodeset
  # del tb.labels  
  # del tb.isbi2nodeset
  # del tb.toisbi
  # del tb.time2labelset
  # del tb.times

  conformTracking(tb)
  addIsbiLabels(tb)

  # With so many attributes holding state about the graph independently it
  # feels awkward and error prone to change all this state. Alternative
  # ideas are 
  #   1. keep minimal state in tb, then regenerate extra items only
  #      when needed (then discard them immediately).
  #   2. Always check for node existence in single place e.g. `pts` before use.
  #   3. keep track of tb.deadnodes
  #   3. Enforce (2) by making nodes references... 
  #   4. keep state around, but discard it once it becomes invalid! (del tb.pts)
  #   5. 


# ltps: list of pts for each time
# aniso: the pixel/voxel size (relative)
# dub: distance upper bound for child-parent connections (in pixels)
# When a node (time,label) has no parent we map it to (time-1, 0)
# i.e. label zero serves as the background label
# We use the index into ltps as the initial label, but convert
# this label to the ISBI scheme for most processing
def nn_tracking(*,dtps,aniso=(1,1),dub=100):
  parents = dict()
  pts_dict = dict()

  if type(dtps) is list:
    dtps = {i:v for i,v in enumerate(dtps)}

  times = sorted(list(dtps.keys()))

  # put all the points in a big dictionary
  for t,pts in dtps.items(): 
    for i,p in enumerate(pts):
      pts_dict[(t,i)] = p

  t0 = times[0]
  for idx, _ in enumerate(dtps[t0]): parents[(t0,idx)] = None


  # ipdb.set_trace()

  # for each pts(t) connect all the points to nearest neib
  # in the previous pts(t-1).
  for t_idx in range(1,len(times)):
    t = times[t_idx]
    pts = dtps[t]
    t_prev = times[t_idx-1]
    # WARN: do we want t_idx-1 or t-1 ? t_idx-1 connects across many frames.
    pts_prev = dtps[t_prev]

    kdt = KDTree(array(pts_prev)*aniso)
    dist, curr2prev = kdt.query(array(pts)*aniso, k=1, distance_upper_bound=dub)
    for idx_curr,idx_prev in enumerate(curr2prev):
      p = None if idx_prev==len(pts_prev) else (t_prev,idx_prev)
      parents[(t,idx_curr)] = p

  tb = SN(parents=parents, pts=pts_dict)
  conformTracking(tb)
  return tb

# Variant of cpnet.createTarget()
# tb: TrueBranching
# time: int
# img_shape: tuple
# sigmas: tuple of radii for label marker
def createTargetWithTrackingLabels(tb, time, img_shape, sigmas):
  s  = array(sigmas).astype(float)
  ks = floor(7*s).astype(int)   # extend support to 7/2 sigma in every direc
  ks = ks - ks%2 + 1            # enfore ODD shape so kernel is centered! 

  # pts = array(pts).astype(int)
  # FIXME
  nodes = tb.time2labelset[time]

  # lab = array([l for n,l in tb.toisbi.items() if n[0]==time])
  # pts = array([p for n,p in tb.pts.items() if n[0]==time])
  pts = array([tb.pts[(time,l)] for l in nodes])
  lab = [tb.toisbi[(time,l)] for l in nodes]

  if len(pts)==0: return zeros(img_shape).astype(np.int64)

  # create a single kernel patch
  def f(x):
    x = x - (ks-1)/2
    return (x*x/s/s).sum() <= 1 # radius squared
    # return exp(-(x*x/s/s).sum()/2)
  kern = array([f(x) for x in indices(ks).reshape((len(ks),-1)).T]).reshape(ks)
    
  target = zeros(ks + img_shape).astype(np.int64) # include border padding
  w = ks//2                      # center coordinate of kernel
  pts_offset = pts + w           # offset by padding

  for i,p in enumerate(pts_offset):
    target_slice = tuple(slice(a,b+1) for a,b in zip(p-w,p+w))
    target[target_slice] = maximum(target[target_slice], kern*lab[i]) # `maximum` puts labels on top of background (zero)

  remove_pad = tuple(slice(a,a+b) for a,b in zip(w,img_shape))
  target = target[remove_pad]

  return target

# draw tails on centerpoints to show cell motion
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

# Classic Bresenham Algorithm
# Draw a line from pt -> par 
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

# directory: path of e.g. 01_GT/TRA/
def loadISBITrackingFromDisk(directory):
  # WARN: must cast to int explicitly, because type(np.uint + int) is numpy.float64 !
  lbep = np.loadtxt(directory + '/man_track.txt').astype(int)
  if lbep.ndim == 1: lbep = lbep.reshape([1,4])

  # Filter to remove mistakes in man_track.txt
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

    # add the node to TB the first time it appears
    # WARNING: Can't simply use b-1 for parent time! There can be gaps in tracks.
    parent[(b,l)] = (mantrackdict[p].e , p) if p!=0 else None #(b-1,p)
    # if (b-1,p) in {(70, 56), (76, 41)}: ipdb.set_trace()

    # for all subsequent times we know the "parent" node exists
    for time in range(b+1, e+1):
      parent[(time,l)] = (time-1,l)

  tb = SN(parents=parent, pts=pts)
  conformTracking(tb)
  return tb

# Plot the lineage tree nodes and links. Use a compressed spatial axis on the
# horizontal and time moves vertically from top to bottom. 
def drawLineageTree(tb, node2label):

  # nodes sorted by position on x axis
  nodes = [x[0] for x in sorted([[k,*v] for k,v in tb.pts.items() if tb.parents[k] is None], key=lambda x: x[-1])]

  cmap = np.random.rand(256,3).clip(min=0.2)
  cmap[0] = (0,0,0)
  cmap = matplotlib.colors.ListedColormap(cmap)

  # map labels to colors
  def l2c(label):
    if label==0:
      return cmap(0)
    else:
      return cmap(label % 254 + 1)

  plt.figure()
  linecollection = mc.LineCollection([], colors='k', linewidths=2)
  plt.gca().add_collection(linecollection)
  lines = []

  # Starting x positions are equally spaced on the X axis. An initial spacing
  # of 3 allow for multiple division before children begin to overlap.
  xpos = {n:3*i for i,n in enumerate(nodes)}

  # As we move through the child nodes we plot them with y positions 
  # given by time and x positions equal to their parents.
  while len(nodes) > 0:

    times = [n[0] for n in nodes]
    xs = [xpos[n] for n in nodes]
    color = [l2c(node2label[n]) for n in nodes]

    plt.scatter(xs,times,c=color)

    # After plotting we replace the list of nodes with all of their children.
    # If there are multiple children, then offset them (asymmetrically) from
    # parent X position.
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

# Add time2labelset and assert that all the nodes
# have an associated spatial point.
def conformTracking(tb):
  d = dict()
  for time,label in tb.pts.keys():
    d[time] = d.get(time, set()) | {label}
  tb.time2labelset = d
  tb.times = sorted(list(tb.time2labelset.keys()))

  # invert parents to get list of children
  tb.children = dict()
  for n,p in tb.parents.items():
    if p is None: continue
    tb.children[p] = tb.children.get(p, []) + [n]

  assert tb.pts.keys() == tb.parents.keys()
  pk = tb.parents.keys()
  # pv = {v for v in tb.parents.values() if v[0]!=-1 and v[1]!=0}
  pv = {v for v in tb.parents.values() if v is not None}
  assert pv-pk==set()

# Adds `time2labelset` to gt and yp.
# Returns global set similarity scores for nodes and edges
# across the entire timeseries
def compare_trackings(gt,yp,aniso,dub):

  times = sorted(list(gt.time2labelset.keys() | yp.time2labelset.keys()))
  
  # maybe find point matches should work with maps instead of arrays...
  # the default should be that we pass around unique id's for pts... i.e. label, (time,label), (gt,time,label), etc
  # an index into a dense array must always be mapped to/from. is always arbitrary. but sometimes all we have!
  def f(t):
    _gt = {(t,l):gt.pts[(t,l)] for l in gt.time2labelset[t]}
    _yp = {(t,l):yp.pts[(t,l)] for l in yp.time2labelset[t]}
    # ipdb.set_trace()
    return snnMatch(_gt, _yp, dub=dub, scale=aniso)
  node_matches = {t:f(t) for t in times}

  # TODO: do more with the scores for all timepoints

  node_totals = build_scores(
    n_m  = sum([x.n_matched for x in node_matches.values()]),
    n_p  = sum([x.n_proposed for x in node_matches.values()]),
    n_gt = sum([x.n_gt for x in node_matches.values()]),
    )

  # count edges. if parent label == 0 => cell appearance
  n_edges_gt = len([v for v in gt.parents.values() if v != None])
  n_edges_yp = len([v for v in yp.parents.values() if v != None])

  # Now we find matching edges by going pt -> parent -> match == pt -> match -> parent (it's a commutative diagram)
  edge_matches = dict()
  # gt_edges_nomatch = set()
  for t in times[1:]:
    for n1,n2 in node_matches[t].matches.items():
      p1 = gt.parents[n1]
      p2 = yp.parents[n2]
      # ipdb.set_trace()
      if p1 is None or p2 is None:
        continue

      # WARN: use p1[0] instead of i-1 because of gaps in tracks!
      if node_matches[p1[0]].matches.get(p1,None)==p2:
        edge_matches[frozenset({p1,n1})] = frozenset({p2,n2})

      # else:
      #   print(f"No matching parents {n1}->{p1} != {n2}->{p2}")
      #   # ipdb.set_trace()
      #   gt_edges_nomatch |= {frozenset({p1,n1})}

  edge_totals = build_scores(n_m=len(edge_matches), n_p=n_edges_yp, n_gt=n_edges_gt)

  return SN(node=node_totals, edge=edge_totals)






