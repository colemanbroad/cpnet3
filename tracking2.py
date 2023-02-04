import numpy as np
from pointmatch import snnMatch
import networkx as nx
import ipdb
from types import SimpleNamespace
from pykdtree.kdtree import KDTree
import matplotlib

from numpy import array, exp, zeros, maximum, indices
def ceil(x): return np.ceil(x).astype(int)
def floor(x): return np.floor(x).astype(int)





## parent_list to TrueBranching (NetworkX tracking datastructure)
def _parent_list_to_tb(parents,ltps):
  """
  parents == -1 when no parent exists
  layer == -2 when no point exists
  This representation cannot describe jumps/gaps in tracking.
  """
  list_of_edges = []
  for time,layer in enumerate(parents):
    if layer is []: continue
    list_of_edges.extend([((time+1,n),(time,parent_id)) for n,parent_id in enumerate(layer) if parent_id!=-1])
  tb = nx.from_edgelist(list_of_edges, nx.DiGraph())
  tb = tb.reverse()

  all_nodes = [(_time,idx) for _time in np.r_[:len(ltps)] for idx in np.r_[:len(ltps[_time])]]
  tb.add_nodes_from(all_nodes)

  for v in tb.nodes: tb.nodes[v]['time'] = v[0]
  for v in tb.nodes: tb.nodes[v]['pt'] = ltps[v[0]][v[1]]

  ## ISBI Labels for each node by incrementing a global track_id
  track_id = 1
  ## first, find all tree roots
  source = [n for n,d in tb.in_degree if d==0]
  ## for every root
  for s in source:
    ## label each node in the tree; when cells divide, inc the track_id;
    ## preorder traversal is 1. root, 2. left subtree, 3. right subtree
    for v in nx.dfs_preorder_nodes(tb,source=s):
      tb.nodes[v]['track'] = track_id
      tb.nodes[v]['root']  = s
      if tb.out_degree[v] != 1: track_id+=1

  return tb

## requires: ltps -- List of Time Points (list of NxD nd-arrays)
## produces: tb -- TrueBranching (directed graph of cell lineage)
def nn_tracking(ltps=None, aniso=(1,1,1), dub=None):
  """
  ltps should be time-ordered list of ndarrays w shape (N,M) with M in [2,3].
  x[t+1] matches to first nearest neib of x[t].

  points can be indexed by (time,local index ∈ [0,N_t])
  """

  dists, parents = [],[]

  for i in range(len(ltps)-1):
    Nparents = len(ltps[i])
    N = len(ltps[i+1])
    if Nparents==0:
      dists.append(np.full(N, -1))
      parents.append(np.full(N, -1))
      continue
    if N==0:
      dists.append([])
      parents.append([])
      continue

    kdt = KDTree(ltps[i]*aniso)
    _dis, _ind = kdt.query(ltps[i+1]*aniso, k=1, distance_upper_bound=dub)
    _ind = _ind.astype(np.int32)
    # if (_ind==8).sum()>0: ipdb.set_trace()
    _ind[_ind==Nparents] = -1
    dists.append(_dis)
    parents.append(_ind)

  # ipdb.set_trace()
  tb = _parent_list_to_tb(parents,ltps)

  return tb

## tb : TrueBranching
## time : int
## rawshape : tuple of [Z]YX dims 
## halfwidth : length of side of square(cube) is 2*halfwidth + 1
def make_ISBI_label_img(tb, time, rawshape, halfwidth=6):
  assert len(rawshape) in (2,3)
  rawshape = np.array(rawshape)
  mantrack = np.zeros(rawshape, np.uint32)

  lab = np.array([tb.nodes[n]['track'] for n in tb.nodes if n[0]==time])
  pts = np.array([tb.nodes[n]['pt'] for n in tb.nodes if n[0]==time])

  h = halfwidth
  for dx in np.arange(-h,h+1):
    for dy in np.arange(-h,h+1):
      p = (pts + (dx,dy)).clip(min=(0,0), max=rawshape-(1,1))
      mantrack[tuple(p.T)] = lab
  return mantrack

## Variant of cpnet.createTarget()
## tb: TrueBranching
## time: int
## img_shape: tuple
## sigmas: tuple of radii for label marker
def createTarget(tb, time, img_shape, sigmas):
  s  = np.array(sigmas)
  ks = floor(7*s).astype(int)   ## extend support to 7/2 sigma in every direc
  ks = ks - ks%2 + 1            ## enfore ODD shape so kernel is centered! 

  # pts = np.array(pts).astype(int)
  lab = np.array([tb.nodes[n]['track'] for n in tb.nodes if n[0]==time])
  pts = np.array([tb.nodes[n]['pt'] for n in tb.nodes if n[0]==time])

  ## create a single Gaussian kernel array
  def f(x):
    x = x - (ks-1)/2
    return (x*x/s/s).sum() < 1
    # return exp(-(x*x/s/s).sum()/2)
  kern = array([f(x) for x in indices(ks).reshape((len(ks),-1)).T]).reshape(ks)
  
  target = zeros(ks + img_shape) ## include border padding
  w = ks//2                         ## center coordinate of kernel
  pts_offset = pts + w              ## offset by padding

  for i,p in enumerate(pts_offset):
    target_slice = tuple(slice(a,b+1) for a,b in zip(p-w,p+w))
    target[target_slice] = kern*lab[i]
    # target[target_slice] = maximum(target[target_slice], kern)

  remove_pad = tuple(slice(a,a+b) for a,b in zip(w,img_shape))
  target = target[remove_pad]

  return target

## plot in matplotlib with color cell lineage
def draw(tb):
  pos  = nx.multipartite_layout(tb,subset_key='time')
  cmap = matplotlib.colors.ListedColormap(np.random.rand(256,3))
  # colors = np.array([tb.edges[e]['track'] for e in tb.edges])
  colors = np.array([tb.nodes[n]['track'] for n in tb.nodes])
  nx.draw(tb,pos=pos,node_size=30,node_color=cmap(colors))

## keep only every nth timepoint, but maintain cell lineage
def subsample_graph(tb,subsample=2):
  edges = []
  nodes = np.array(tb.nodes)
  tmax = nodes[:,0].max()
  newnodes = []

  for t in range(0,tmax+1,subsample):
    tnodes = nodes[nodes[:,0]==t]
    for n in tnodes:
      newnodes.append(n)
      n = tuple(n)
      cn = n    ## current node
      count = 0 ## how many parents have we climbed
      while True:
        l = list(tb.pred[cn])
        if len(l)==0: break
        elif (count==subsample-1):
          edges.append((l[0] , n))
          break
        else:
          cn = l[0]
          count += 1

  s = subsample
  edges_newtime = [((int(a[0]/s),a[1]) , (int(b[0]/s),b[1])) for a,b in edges]
  tbnew = nx.from_edgelist(edges_newtime , create_using=nx.DiGraph)
  newnodes_newtime = [(int(a[0]/s),a[1]) for a in newnodes]
  tbnew.add_nodes_from(newnodes_newtime)

  for n in tbnew.nodes:
    tbnew.nodes[n]['pt'] = tb.nodes[(int(n[0]*s),n[1])]['pt']
  return tbnew

## Given two TB's we want to compute TP/FP/FN for detections and edges across the whole timeseries.
## Also, we should compute some stats about the length of valid tracking trajectories.
def compare_trackings(tb_gt,tb,aniso=[1,1],dub=20):

  times = sorted(list(set([n[0] for n in tb_gt.nodes])))

  ## First match detections across all time
  ## `pts` are (N,D) arrays of [2,3]D coordinates. `lab` is (N,) array storing labels for `pts`.
  matches = dict()
  for i in times:

    T = SimpleNamespace()
    T.lab0 = np.array([n[1] for n in tb_gt.nodes if n[0]==i])
    pts0   = np.array([tb_gt.nodes[n]['pt'] for n in tb_gt.nodes if n[0]==i])
    T.lab1 = np.array([n[1] for n in tb.nodes if n[0]==i])
    pts1   = np.array([tb.nodes[n]['pt'] for n in tb.nodes if n[0]==i])

    T.match = snnMatch(pts0,pts1,dub=dub,aniso=aniso)
    matches[i] = T
    # print(f"T={i} P={T.match.precision:.5f} R={T.match.recall:.5f} F1={T.match.f1:.5f}")
    # ipdb.set_trace()

  ## Second, match on edges.
  ## Iterate over edges. first GT, then proposed.
  ## for each edge get parent -> match1 , child -> match2. assert (match1,match2) in proposed edges.
  edges_gt = np.zeros(len(tb_gt.edges))
  for n,e in enumerate(tb_gt.edges):
    t0 = e[0][0]
    t1 = e[1][0]
    ## get the index of e[0] and e[1]
    idx0 = np.argwhere(matches[t0].lab0==e[0][1])[0,0]
    idx1 = np.argwhere(matches[t1].lab0==e[1][1])[0,0]

    ## do the detection matches exist?
    if matches[t0].match.n_matched==0 and matches[t1].match.n_matched==0:
      edges_gt[n]=5
      continue
    elif matches[t0].match.n_matched==0:
      edges_gt[n]=3
      continue
    elif matches[t1].match.n_matched==0:
      edges_gt[n]=4
      continue

    matched_0 = matches[t0].match.gt_matched_mask[idx0]
    matched_1 = matches[t1].match.gt_matched_mask[idx1]

    ## if they both exist, then do those matches share an edge?
    if matched_0 and matched_1:
      l0 = matches[t0].lab1[matches[t0].match.gt2yp[idx0]]
      l1 = matches[t1].lab1[matches[t1].match.gt2yp[idx1]]
      e2 = ((t0,l0) , (t1,l1))
      matched_edge = tb.edges.get(e2 , None) ## None is default, but we are explicit

      ## woohoo! A match!
      if matched_edge is not None: ## must be explicit, because default value is empty set
        edges_gt[n]=1 ## maybe use (e,e2) for more info later

      ## out edge didn't match because of linking problem
      else:
        edges_gt[n]=2

    ## our edge didn't match because of mis-detection.
    elif matched_0:
      edges_gt[n]=3
    elif matched_1:
      edges_gt[n]=4
    else:
      edges_gt[n]=5 ## two mis-detections


  ## NOW do the same iteration again, but over the proposed (tb, lab1, pts1, etc)
  edges_prop = np.zeros(len(tb.edges))
  for n,e in enumerate(tb.edges):
    t0 = e[0][0]
    t1 = e[1][0]
    ## get the index of e[0] and e[1]
    idx0 = np.argwhere(matches[t0].lab1==e[0][1])[0,0]
    idx1 = np.argwhere(matches[t1].lab1==e[1][1])[0,0]
    ## do the detection matches exist?
    matched_0 = matches[t0].match.yp_matched_mask[idx0]
    matched_1 = matches[t1].match.yp_matched_mask[idx1]

    ## if they both exist, then do those matches share an edge?
    if matched_0 and matched_1:
      l0 = matches[t0].lab0[matches[t0].match.yp2gt[idx0]]
      l1 = matches[t1].lab0[matches[t1].match.yp2gt[idx1]]
      e2 = ((t0,l0) , (t1,l1))
      matched_edge = tb_gt.edges.get(e2 , None) ## None is default, but we are explicit

      ## woohoo! A match!
      if matched_edge is not None:
        edges_prop[n]=1 ## maybe use (e,e2) for more info later

      ## out edge didn't match because of linking problem
      else:
        edges_prop[n]=2

    ## our edge didn't match because of mis-detection.
    elif matched_0:
      edges_prop[n]=3
    elif matched_1:
      edges_prop[n]=4
    else:
      edges_prop[n]=5 ## two mis-detections


  assert (edges_prop==1).sum() == (edges_gt==1).sum()

  ## compute all the interesting whole-movie statistics about TP/FN/FP det's and edges
  S = SimpleNamespace() ## Scores

  ## detections

  S.n_matched_det  = sum([t.match.n_matched  for t in matches.values()])
  S.n_gt_det       = sum([t.match.n_gt       for t in matches.values()])
  S.n_proposed_det = sum([t.match.n_proposed for t in matches.values()]) ## n_proposed
  S.precision_det  = S.n_matched_det   / S.n_proposed_det
  S.recall_det     = S.n_matched_det   / S.n_gt_det
  S.f1_det         = S.n_matched_det*2 / (S.n_gt_det + S.n_proposed_det)

  ## edges

  S.n_matched_tra  = (edges_gt==1).sum() ## equiv to (edges_prop==1).sum()
  S.n_gt_tra       = edges_gt.shape[0]
  S.n_proposed_tra = edges_prop.shape[0]
  S.precision_tra  = S.n_matched_tra   / S.n_proposed_tra
  S.recall_tra     = S.n_matched_tra   / S.n_gt_tra
  S.f1_tra         = S.n_matched_tra*2 / (S.n_gt_tra + S.n_proposed_tra)

  # for k,v in S.__dict__.items():
  #   print(f"{k:12s}\t{v:.5f}")

  return matches , edges_gt , edges_prop , S








