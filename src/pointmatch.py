from pykdtree.kdtree import KDTree
import numpy as np
from types import SimpleNamespace
import ipdb

def snnMatch(gt__pt,yp__pt,dub=10,scale=[1,1,1]):
  
  res = SimpleNamespace()
  if type(gt__pt) is not dict:
    gt__pt = {i:v for i,v in enumerate(gt__pt)}
  if type(yp__pt) is not dict:
    yp__pt = {i:v for i,v in enumerate(yp__pt)}

  n_p  = len(yp__pt)
  n_gt = len(gt__pt)

  if n_p==0 or n_gt==0:
    return SimpleNamespace(n_matched=0, n_proposed=n_p, n_gt=n_gt, precision=0, f1=0, recall=0)

  gt = SimpleNamespace()
  gt.l2pt = gt__pt  ## label to point
  gt.i2pt = []      ## index to point
  gt.i2l  = dict()  ## index to label

  yp = SimpleNamespace()
  yp.l2pt = yp__pt
  yp.i2pt = []
  yp.i2l  = dict()

  for i,(l,pt) in enumerate(gt.l2pt.items()):
    gt.i2pt.append(pt)
    gt.i2l[i] = l

  for i,(l,pt) in enumerate(yp.l2pt.items()):
    yp.i2pt.append(pt)
    yp.i2l[i] = l

  ## scale points to compensate for anisotropy
  gt.i2pt = np.array(gt.i2pt) * scale
  yp.i2pt = np.array(yp.i2pt) * scale
  kdt = KDTree(yp.i2pt)
  gt.dist, gt.i2i = kdt.query(gt.i2pt , k=1, distance_upper_bound=dub)
  kdt = KDTree(gt.i2pt)
  yp.dist, yp.i2i = kdt.query(yp.i2pt , k=1, distance_upper_bound=dub)

  ## convert indices back to labels
  gt.l2l = {gt.i2l[i]:yp.i2l.get(gt.i2i[i],None) for i in range(n_gt)}
  yp.l2l = {yp.i2l[i]:gt.i2l.get(yp.i2i[i],None) for i in range(n_p)}

  ## only need to determine the loop from one side so choose gt.l2l .
  matching = dict()
  for k,v in gt.l2l.items():
    if k==yp.l2l.get(v,None):
      # ipdb.set_trace()
      matching[k] = v

  # res.matches = matching
  res = build_scores(n_m=len(matching), n_p=n_p, n_gt=n_gt)
  res.matches = matching
  return res

def build_scores(*,n_m, n_p, n_gt, return_dict=False):

  if n_p==0 or n_gt==0:
    return SimpleNamespace(n_matched=0, n_proposed=n_p, n_gt=n_gt, precision=0, f1=0, recall=0)

  res = SimpleNamespace()
  res.n_matched  = n_m
  res.n_proposed = n_p
  res.n_gt       = n_gt
  res.precision  = n_m / n_p
  res.recall     = n_m / n_gt
  res.f1         = 2*n_m / (n_p + n_gt)
  if return_dict: return res.__dict__
  return res

