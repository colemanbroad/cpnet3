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
  plt.figure()
  for i,_ in enumerate(list_of_pointclouds):
    if i==0: continue

    # if p[t] has a sibling or no parent: assign new ID. 
    # otherwise inherit ID from parent
    for j,_ in enumerate(labels[i]):
      assert edges[i-1][:,j].sum(0) == 1
      parent_id = edges[i-1][:,j].argmax()
      if edges[i-1][parent_id,:].sum()==1: ## j is only child
        labels[i][j] = labels[i-1][parent_id]
        plt.plot([i-1, i], [x[i-1][parent_id][0], x[i][j][0]])
      else:
        labels[i][j] = currentmax+1
        currentmax += 1
        plt.plot(i,x[i][j][0],'o')

  return edges, labels

def testTrackAndLabel():
  N = 10
  T = 10
  pts = rand(N,1)*20
  lpts = [pts + rand(N,1) for _ in range(T)]
  edges, labels = trackAndLabel(lpts)
  # print(edges,labels)

## Minimum Euclidean Distance Tracking with max-two-daughters constraint.
## 140ms (20ms) for 100 x 100 pts (@jit)
## 17s (7.2s) for 1000 pts (@jit) and
# @jit(nopython=True)
def minCostTrackWithDivisions(pts0,pts1):
  N = pts0.shape[0]
  M = pts1.shape[0]
  costs = np.zeros((N,M),np.float32)
  edges = np.zeros((N,M),np.uint8)
  for j in range(N):
    for k in range(M):
      costs[j,k] = np.linalg.norm(pts0[j]-pts1[k], ord=2) # euclidean

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
      costs[p,c]=np.inf

    # ipdb.set_trace()
    ## if every child has a parent, break
    if (s0==0).sum()==0: break


  print(edges.sum(1))
  if np.any(edges.sum(0)!=1): ipdb.set_trace()

  return edges


def testMinCostTrackWithDivisions(N=10, D=1, b=2):
  # ltps = [rand(randint(50,70)) for _ in range(10)]
  # print(trackAndLabel(ltps))
  pts0  = rand(N,D) * b
  pts1  = pts0 + rand(N,D)
  pts2  = cat( [pts0 + rand(N,D) , rand(N//5,D)*b], axis=0)
  edges = minCostTrackWithDivisions(pts0,pts1)
  plt.imshow(edges)

def makeISBILabels(list_of_pointclouds, rawshape):
  assert len(rawshape) in (2,3)
  edges, labels = trackAndLabel(list_of_pointclouds)
  mantrack_list = []
  for i,lab in enumerate(labels):
    mantrack = np.zeros(rawshape, np.uint32)
    pts = list_of_pointclouds[i]
    for dx in np.arange(-6,6):
      for dy in np.arange(-6,6):
        mantrack[tuple((pts + (dx,dy)).T)] = labels[i]
    mantrack_list.append(mantrack)
    # ipdb.set_trace()
  return mantrack_list
  # if mantrack.ndim == 3: mantrack = mantrack.sum(0)
