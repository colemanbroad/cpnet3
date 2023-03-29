from numba import jit
import ipdb
import numpy as np
from numpy.random import rand,randint
from numpy import concatenate as cat

from skimage.io import imsave
import matplotlib.pyplot as plt
plt.ion()

## track points and assign ISBI labels
def trackAndLabel(list_of_pointclouds):
  x = list_of_pointclouds
  edges = [minCostTrackWithDivisions(x[i], x[i+1]) for i in range(len(x)-1)]
  labels = [np.zeros(xi.shape[0],np.uint32) for xi in x]
  labels[0] = np.arange(labels[0].shape[0]) + 1
  currentmax = np.max(labels[0])
  # f1,ax1 = plt.subplots()
  # f2,ax2 = plt.subplots()

  for i, xi in enumerate(x):
    for j, xij in enumerate(xi):
      # ax1.plot(i, xij[0], 'o', color=plt.cm.Set1(j%9))
      # ax1.plot(xij[0], xij[1], 'o', color=plt.cm.Set1(j%9))
      pass

  for t in range(1,len(x)):
    for idx  in range(len(x[t])):

    # if p[t] has a sibling or no parent: assign new ID. 
    # otherwise inherit ID from parent
      assert edges[t-1][:,idx].sum(0) == 1
      parent_idx = edges[t-1][:,idx].argmax()
      parent_pt = x[t-1][parent_idx]
      child_pt = x[t][idx]
      if edges[t-1][parent_idx,:].sum() == 1: ## ## == 1 idx is only child
        labels[t][idx] = labels[t-1][parent_idx]
        # ax1.plot([t-1, t], [x[t-1][parent_idx][0], x[t][idx][0]], 'k')
        # ax1.plot([parent_pt[0], child_pt[0]], [parent_pt[1], child_pt[1]], 'k')
      else:
        labels[t][idx] = currentmax+1
        currentmax += 1
        # ax1.plot([t-1, t], [x[t-1][parent_idx][0], x[t][idx][0]], 'k')
        # ax1.plot([parent_pt[0], child_pt[0]], [parent_pt[1], child_pt[1]], 'k')
      # ax2.plot([t-1, t], [x[t-1][parent_idx][0], x[t][idx][0]], 'o-', color=plt.cm.Set1(labels[t][idx]%9))
      # ax2.plot([parent_pt[0], child_pt[0]], [parent_pt[1], child_pt[1]], 'o-', color=plt.cm.Set1(labels[t][idx]%9))

      # print(t, idx , labels[t][idx])

  return edges, labels

def testTrackAndLabel():
  N = 10
  T = 10
  pts = rand(N,2)*5
  lpts = [pts + rand(N,2) for _ in range(T)]
  edges, labels = trackAndLabel(lpts)
  # print(edges,labels)

## Minimum Euclidean Distance Tracking with max-two-daughters constraint.
## 140ms (20ms) for 100 x 100 pts (@jit)
## 17s (7.2s) for 1000 pts (@jit) and
# @jit(nopython=True)
def minCostTrackWithDivisions(pts0,pts1):
  N = pts0.shape[0]
  M = pts1.shape[0]
  pts0 = pts0.astype(np.float32)
  pts1 = pts1.astype(np.float32)

  costs = np.zeros((N,M),np.float32)
  edges = np.zeros((N,M),np.uint8)
  for j in range(N):
    for k in range(M):
      costs[j,k] = np.linalg.norm(pts0[j]-pts1[k], ord=2) # euclidean
      # ipdb.set_trace()

  # plt.figure()
  # plt.imshow(costs)
  while True:
    min_cost = costs.min()
    min_idx  = costs.argmin()
    p,c = min_idx//M, min_idx%M # [p]arent, [c]hild
    s0 = edges.sum(0)
    s1 = edges.sum(1)

    if min_cost==np.inf: break

    ## if the parent has 0 or 1 daughter, add this daughter
    ## otherwise remove this parent as an option
    # if parent has fewer < 2 children
    if (s1[p]<2):
      edges[p,c]=1
      costs[:,c]=np.inf
    else:
      costs[p,:]=np.inf

    # ipdb.set_trace()
    ## if every child has a parent, break
    if (s0==0).sum()==0: break

  # plt.figure()
  # plt.imshow(edges)

  # print(edges.sum(1))
  if np.any(edges.sum(0)!=1): ipdb.set_trace()

  return edges

def testMinCostTrackWithDivisions(N=10, D=1, b=2):
  # ltps = [rand(randint(50,70)) for _ in range(10)]
  # print(trackAndLabel(ltps))
  pts0  = rand(N,D) * b
  pts1  = pts0 + rand(N,D)
  pts2  = cat( [pts0 + rand(N,D) , rand(N//5,D)*b], axis=0)
  edges = minCostTrackWithDivisions(pts0,pts1)
  # plt.imshow(edges)

def make_ISBI_label_img(pts, labels, rawshape):
  assert len(rawshape) in (2,3)
  rawshape = np.array(rawshape)
  mantrack = np.zeros(rawshape, np.uint32)
  for dx in np.arange(-6,6):
    for dy in np.arange(-6,6):
      p = (pts + (dx,dy)).clip(min=(0,0), max=rawshape-(1,1))
      mantrack[tuple(p.T)] = list_of_labels[i]
  return mantrack

  # list_of_edges, list_of_labels = trackAndLabel(list_of_pointclouds)
  # mantrack_list = []
  # s1,s2 = set(),set()
  # for i,_ in enumerate(list_of_labels):
  #   mantrack = np.zeros(rawshape, np.uint32)
  #   pts = list_of_pointclouds[i]
  #   for dx in np.arange(-6,6):
  #     for dy in np.arange(-6,6):
  #       p = (pts + (dx,dy)).clip(min=(0,0), max=rawshape-(1,1))
  #       mantrack[tuple(p.T)] = list_of_labels[i]
  #   mantrack_list.append(mantrack)
  #   # ipdb.set_trace()
  # return mantrack_list
  # if mantrack.ndim == 3: mantrack = mantrack.sum(0)




