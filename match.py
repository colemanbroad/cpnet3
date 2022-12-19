from pykdtree.kdtree import KDTree
import numpy as np
from types import SimpleNamespace

def snnMatch(_pts_gt,_pts_yp,dub=10,scale=[1,1,1]):
  """
  pts_gt is ground truth. pts_yp as predictions. this function is not symmetric!
  we return binary masks for pts_gt and pts_yp where masked elements are matched.
  we also return a mapping from gt2yp and yp2gt matching indices.
  
  We can obtain a matching between points in many different ways.
  The old way was to only compute the function from gt to yp.
  Thus an individual yp may appear zero or more times.
  
  BUG: we never established a true matching, just the score.
  Our scheme was such that all gt points within `dub` distance of a yp were considered matched, even if that match was not unique.
  This makes sense if the matching region associated with a nucleus is _uniquely_ claimed by that nucleus (regions don't overlap).
  If regions _do_ overlap, then this second criterion is active (nucleus center is nearest neib of proposed point).
  We could solve an assignment problem with Hungarian matching to enable even more flexible matching.
  This is only necessary if we have overlapping regions, and it might be possible that proposed point X matches to gt point Y1 even though it is closer to Y2.

  Tue Apr 13 13:35:02 2021
  Return nan for precision|recall|f1 when number of objects is zero
  """

  res = SimpleNamespace()

  def _final_scores(n_m,n_p,n_gt):
    with np.errstate(divide='ignore',invalid='ignore'):
      precision = np.divide(n_m  ,  n_p)
      recall    = np.divide(n_m  ,  n_gt)
      f1        = np.divide(2*n_m,  (n_p + n_gt))

    res = SimpleNamespace()
    res.n_matched = n_m
    res.n_proposed = n_p
    res.n_gt = n_gt
    res.precision = precision
    res.f1 = f1
    res.recall = recall
    return res

  if len(_pts_gt)==0 or len(_pts_yp)==0:
    n_matched  = 0
    n_proposed = len(_pts_yp)
    n_gt       = len(_pts_gt)
    return _final_scores(n_matched,n_proposed,n_gt)

  pts_gt = np.array(_pts_gt) * scale ## for matching in anisotropic spaces
  pts_yp = np.array(_pts_yp) * scale ## for matching in anisotropic spaces

  kdt = KDTree(pts_yp)
  gt2yp_dists, gt2yp = kdt.query(pts_gt, k=1, distance_upper_bound=dub)
  kdt = KDTree(pts_gt)
  yp2gt_dists, yp2gt = kdt.query(pts_yp, k=1, distance_upper_bound=dub)

  N = len(pts_gt)
  inds = np.arange(N)
  ## must extend yp2gt with N for gt points whose nearest neib is beyond dub
  res.gt_matched_mask = np.r_[yp2gt,N][gt2yp]==inds
  N = len(pts_yp)
  inds = np.arange(N)
  res.yp_matched_mask = np.r_[gt2yp,N][yp2gt]==inds

  assert res.gt_matched_mask.sum() == res.yp_matched_mask.sum()
  res.gt2yp = gt2yp
  res.yp2gt = yp2gt
  res.pts_gt = _pts_gt ## take normal points, not rescaled!
  res.pts_yp = _pts_yp ## take normal points, not rescaled!

  res.n_matched  = res.gt_matched_mask.sum()
  res.n_proposed = len(pts_yp)
  res.n_gt       = len(pts_gt)
  res.precision  = res.n_matched / res.n_proposed
  res.recall     = res.n_matched / res.n_gt
  res.f1         = 2*res.n_matched / (res.n_proposed + res.n_gt)

  return res