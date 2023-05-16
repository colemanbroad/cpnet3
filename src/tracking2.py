import numpy as np
import ipdb
from types import SimpleNamespace as SN
from pykdtree.kdtree import KDTree
import matplotlib
from pointmatch import snnMatch, build_scores
from cpnet import load_isbi_csv
import os
import pickle

from collections import defaultdict
from numba import jit
from scipy.optimize import linear_sum_assignment
import matplotlib.collections as mc
import matplotlib.pyplot as plt
from random import random

from tifffile                  import imread
import re
from skimage.measure           import regionprops
from os import path
from tabulate import tabulate

from glob import glob

from numpy import array, exp, zeros, maximum, indices, any, all
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

  labels = dict()
  currentmax = 1
  for t in tb.time2labelset.keys():
    for idx in tb.time2labelset[t]:

      node = (t,idx)
      # if node has no parent or does have siblings: assign new ID. 
      parent = tb.parents[node]
      if parent is None or len(tb.children[parent])>1:
        labels[node] = currentmax

        currentmax += 1
        continue

      # otherwise inherit ID from parent
      labels[node] = labels[parent]


  tb.toisbi = labels


# Classify all nodes into 'enter', 'exit', and 'move' and classify all links
# into 'move' and 'divide'.
def addNodeClassification(tb):

  labels = dict()
  edge_labels = dict()

  for t in tb.time2labelset.keys():
    for idx in tb.time2labelset[t]:

      node = (t,idx)
      parent = tb.parents[node]
      children = tb.children.get(node,None)

      if parent is None:
        # if node has no parent it's an enter
        labels[node] = 'enter'
      elif children is None:
        # if node has no children it's an exit
        labels[node] = 'exit'
      else:
        # otherwise it's a move
        labels[node] = 'move'

      if parent is None: continue

      if len(tb.children[parent])==1:
        # if node has no siblings it's link to parent is a movement
        edge_labels[node] = 'move'
      else:
        # otherwise it's got siblings and it's link to parent is a division
        edge_labels[node] = 'divide'

  tb.labels = labels
  tb.edge_labels = edge_labels

def remNodeClassification(tb):
  del tb.labels
  del tb.edge_labels

def groupby(dikt):
  inverse = defaultdict(set)
  for key,val in dikt.items():
    inverse[val].add(key)
  return inverse

# Remove singleton branches which are usually false divisions.
def pruneSingletonBranches(tb):

  ## groupby toisbi
  isbi2nodeset = groupby(tb.toisbi)

  # Singleton branches have no children and either:
  #  no parent. The node appears and then dies.
  #  yes parent. The node is likely a false division resulting from FP detection.
  for (k,ns) in isbi2nodeset.items():
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
    if parent is not None:
      tb.children[parent].remove(node) ## in-place
    
    continue

    ## if we want to try and fix the isbi labeling in place instead of 
    ## regenerating it we can do this... Otherwise regen everything.
    if len(tb.children[parent]) != 1: continue
    
    onlychild = list(tb.children[parent])[0]
    child_label = tb.toisbi[onlychild]
    parent_label = tb.toisbi[parent]
    for n in isbi2nodeset[child_label]:
      tb.toisbi[n] = parent_label

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

# # Compensate for drift by 
# def 

# from numba import jit, prange
# @jit(nopython=True, fastmath=True, nogil=True, cache=True, parallel=True)
# def interpolate_bilinear(array_in, width_in, height_in, array_out, width_out, height_out):
#     for i in prange(height_out):
#         for j in prange(width_out):



# ltps: list of pts for each time
# aniso: the pixel/voxel size (relative)
# dub: distance upper bound for child-parent connections (in pixels)
# When a node (time,label) has no parent we map it to (time-1, 0)
# i.e. label zero serves as the background label
# We use the index into ltps as the initial label, but convert
# this label to the ISBI scheme for most processing
def link_nearestNeib(*,dtps,aniso=(1,1),dub=100):
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


# Minimum Euclidean Distance Tracking with max-two-daughters constraint.
# 140ms (20ms) for 100 x 100 pts (@jit)
# 17s (7.2s) for 1000 pts (@jit) and
@jit(nopython=True)
def greedyMinCostAssignment(costs, max_children=2):

  # determine edges by greedily minimizing costs
  N,M = costs.shape
  edges = np.zeros((N,M),dtype=np.uint8)
  while True:
    min_cost = costs.min()
    min_idx  = costs.argmin()
    p,c = min_idx//M, min_idx%M # [p]arent, [c]hild

    # if the lowest cost is infinite, break
    if min_cost==np.inf: break

    # if every child has a parent, break
    if all(edges.sum(0)==1): break

    # If the parent has 0 or 1 daughter, add (p,c) to edge set. Otherwise set
    # this parent's edge costs to inf.
    if (edges[p].sum() < max_children):
      edges[p,c]=1
      costs[:,c]=np.inf
    else:
      costs[p,:]=np.inf

  assert all(edges.sum(0) <= 1)

  return edges

@jit(nopython=True)
def costmatrix(pts0, pts1, dub=100):
  N,M = len(pts0),len(pts1)
  costs = np.zeros((N,M),np.float32)
  for j in range(N):
    for k in range(M):
      dist = np.linalg.norm(pts0[j]-pts1[k], ord=2)
      costs[j,k] = dist if dist < dub else 100
  return costs

# We can use arbitrary costs, not just euclidean distance.
# Track by greedily choosing assignments that minimize those costs
# without violating the max-two-daughters constraint.
def link_minCostAssign(*,dtps,aniso=(1,1),dub=100, greedy=True):
  parents = dict()
  pts_dict = dict()

  if type(dtps) is list:
    dtps = {i:v for i,v in enumerate(dtps)}

  times = sorted(list(dtps.keys()))

  # put all the points in a big dictionary
  for t,pts in dtps.items():
    for i,p in enumerate(pts):
      pts_dict[(t,i)] = p

  # now rescale the points for tracking
  for k,v in dtps.items():
    dtps[k] = array(v) * aniso

  t0 = times[0]
  for idx, _ in enumerate(dtps[t0]): parents[(t0,idx)] = None

  for i in range(1, len(times)):
    t0 = times[i-1]
    t1 = times[i]
    pts0 = dtps[t0].astype(np.float32)
    pts1 = dtps[t1].astype(np.float32)
    N,M = len(pts0),len(pts1)

    if M==0: continue
    if N==0:
      for i,_ in enumerate(pts1):
        parents[(t1,i)] = None
      continue

    costs = costmatrix(pts0,pts1,dub=dub)

    if greedy:
      edges = greedyMinCostAssignment(costs, max_children=2)
    else:
      # duplicate the cost matrix along the row axis, to there are two copies
      # of every parent node. This allows matching to a parent node twice by
      # matching to it's copy. We make the matrix square by extending the cols
      # (child axis) with dummy variables until it has the same size as rows.

      costs = np.concatenate([costs,costs],axis=0)
      if 2*N-M > 0:
        # add dummy division variables 
        dummy = np.zeros([2*N, 2*N - M]) + 100
        costs = np.concatenate([costs,dummy],axis=1)
      else:
        # we have more than twice pts in t1 as t0! need dummy entrance pts.
        # dummy entrance cost is 100
        dummy = np.zeros([M-2*N, M]) + 100
        costs = np.concatenate([costs,dummy],axis=0)



      row_ind, col_ind = linear_sum_assignment(costs, maximize=False)
      edges = np.zeros([N,M], dtype=np.uint8)
      for r,c in zip(row_ind,col_ind):
        if c>=M: continue
        ## mod M 
        edges[r%N,c] = 1

    parent_id_list = edges.argmax(0)
    for child_id,parent_id in enumerate(parent_id_list):
      parents[(t1,child_id)] = (t0,parent_id)

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
def compare_trackings(gt,yp,aniso,dub,byclass=True):

  res = dict() ## store results

  times = sorted(list(gt.time2labelset.keys() | yp.time2labelset.keys()))

  if byclass:
    addNodeClassification(gt)
    addNodeClassification(yp)
  
  # maybe find point matches should work with maps instead of arrays... the
  # default should be that we pass around unique id's for pts... i.e. label,
  # (time,label), (gt,time,label), etc an index into a dense array must
  # always be mapped to/from. is always arbitrary. but sometimes all we
  # have!
  def f(t):
    _gt = {(t,l):gt.pts[(t,l)] for l in gt.time2labelset[t]}
    _yp = {(t,l):yp.pts[(t,l)] for l in yp.time2labelset[t]}
    # ipdb.set_trace()
    return snnMatch(_gt, _yp, dub=dub, scale=aniso)
  node_matches = {t:f(t) for t in times}

  node_totals = build_scores(
    n_m  = sum([x.n_matched for x in node_matches.values()]),
    n_p  = sum([x.n_proposed for x in node_matches.values()]),
    n_gt = sum([x.n_gt for x in node_matches.values()]),
    return_dict=True,
    )
  res['node'] = node_totals

  ## match by class
  if byclass:
    # node_confusion = defaultdict(lambda : 0)
    c2i = {'enter':0, 'move':1, 'exit':2, 'miss':3}
    node_confusion = np.zeros([4,4], dtype=np.uint32)
    yp_mset = set()
    for gt_node in gt.pts.keys():
      gt_class = gt.labels[gt_node]
      time = gt_node[0]
      yp_node = node_matches[time].matches.get(gt_node, None)
      if yp_node: yp_mset.add(yp_node)
      yp_class = yp.labels.get(yp_node, 'miss')
      node_confusion[ c2i[yp_class] , c2i[gt_class] ] += 1
    for gt_node in yp.pts.keys() - yp_mset:
      node_confusion[ c2i[yp.labels[gt_node]] , c2i['miss'] ] += 1

    res['node_confusion'] = node_confusion

    # ipdb.set_trace()
    # print("Row (dim 0 / slow): YP, Col (dim 1 / fast): GT")
    # print(c2i)
    # node_confusion[3,2]=999
    # print("Nodes")
    # print(node_confusion)

  # count edges. if parent label == 0 => cell appearance
  n_edges_gt = len([v for v in gt.parents.values() if v != None])
  n_edges_yp = len([v for v in yp.parents.values() if v != None])

  # Now we find matching edges by going pt -> parent -> match == pt ->
  # match -> parent (it's a commutative diagram)

  # But we also have to find all the edges! How are we going to do this in a
  # single pass? We can split the nodes up into yp-matched_yp, matched_yp,
  # gt-matched_gt and then count the number of nodes with parents in
  # unmatched yp and gt nodesets, then count the number of matched nodes with
  # matched parents.

  # See fig1

  #   return dict(**locals())
  # def next(D):
  #   globals().update(D)

  edge_matches = dict()
  edgecounts = dict()
  for t in times[1:]:
    counts = SN(tp=0,yp=0,gt=0)

    yp_edges = dict()
    for k in yp.time2labelset[t]:
      n1 = (t,k)
      p1 = yp.parents[n1]
      if p1 is None: continue
      yp_edges[n1] = (n1,p1)

    gt_edges = dict()
    for k in gt.time2labelset[t]:
      n1 = (t,k)
      p1 = gt.parents[n1]
      if p1 is None: continue
      gt_edges[n1] = (n1,p1)
    
    counts.yp = len(yp_edges)
    counts.gt = len(gt_edges)

    # figure out if the n1->n2->p2->p1 forms a full cycle (undirected)
    for n1,p1 in gt_edges.values():
      n2 = node_matches[t].matches.get(n1, None)
      if n2 is None: continue
      n2p2 = yp_edges.get(n2,None)
      if n2p2 is None: continue
      p2 = n2p2[1]
      # ipdb.set_trace()
      if node_matches[p1[0]].matches.get(p1) != p2: continue
      # if p1 != p2: continue
      edge_matches[(n1,p1)] = (n2,p2)
      counts.tp += 1

    # if counts.yp > 20:

    edgecounts[t] = counts

  edge_totals = build_scores(
    n_m=len(edge_matches), n_p=n_edges_yp, n_gt=n_edges_gt,
    return_dict=True)
  res['edge'] = edge_totals

  # To compute the edge scores by time we need to partition the edges by time
  # and count the number of matches (TP), FP and FN.
  res['edge_by_time'] = {
    t : build_scores(n_m=counts.tp, n_p=counts.yp, n_gt=counts.gt)
    for t,counts in edgecounts.items()
  }

  ## match by class on edges
  if byclass:
    # edge_confusion = defaultdict(lambda : 0)
    c2i = {'move':0, 'divide':1, 'miss':2,}
    edge_confusion = np.zeros([3,3], dtype=np.uint32)
    yp_mset = set()
    # for every edge in the ground truth tracking
    for (n1,p1) in gt.parents.items():
      if p1 is None: continue
      # get it's label
      gt_class = gt.edge_labels[n1]
      # and it's match - if it exists
      yp_edge = edge_matches.get((n1,p1), None)
      # if it does exist add it to the matched set
      # NOTE: must be in n1,p1 order to match with yp.parents.items()
      if yp_edge: 
        yp_mset.add(yp_edge)
        # check that edge is in (child,parent) form
        assert yp_edge[0][0] > yp_edge[1][0]
        yp_class = yp.edge_labels[yp_edge[0]]
      else:
        ## miss if yp_node is none
        yp_class = 'miss'
      # and add the (yp_class, gt_class) pair to the matrix
      edge_confusion[ c2i[yp_class] , c2i[gt_class] ] += 1

    for (n1,p1) in set(yp.parents.items()) - yp_mset:
      if p1 is None: continue
      edge_confusion[ c2i[yp.edge_labels[n1]] , c2i['miss'] ] += 1

    res['edge_confusion'] = edge_confusion

  # for v in node_matches.values():
  #   del v.__dict__['matches']

  res['node_timeseries'] = node_matches

  # ipdb.set_trace()

  return res






