from numba import jit
import ipdb
import numpy as np
from numpy.random import rand,randint
from numpy import concatenate as cat
from numpy import array, any, all 

from skimage.io import imsave
import matplotlib.pyplot as plt
plt.ion()

# Track points and assign ISBI labels.
# edges: list of assignment matrices describing all links
def assignIsbiLabels(edges):
  labels = dict()
  currentmax = 1

  for t in range(len(edges)):
    for idx in range(len(x[t])):

      # if p[t] has a sibling or no parent: assign new ID. 
      # otherwise inherit ID from parent
      assert edges[t][:,idx].sum(0) == 1
      parent_idx = edges[t][:,idx].argmax()

      # if parent doesn't 
      # !!!WARN: WIP...
      if edges[t][parent_idx,:].sum() != 1:
        labels[(t+1, idx)] = currentmax
        currentmax += 1
        continue

      labels[(t+1, idx)] = labels[(t, parent_idx)]

  return edges, labels

def testTrackAndLabel():
  N = 10
  T = 10
  pts = rand(N,2)*5
  lpts = [pts + rand(N,2) for _ in range(T)]
  edges, labels = trackAndLabel(lpts)
  # print(edges,labels)

# Minimum Euclidean Distance Tracking with max-two-daughters constraint.
# 140ms (20ms) for 100 x 100 pts (@jit)
# 17s (7.2s) for 1000 pts (@jit) and
# @jit(nopython=True)
def greedyMinCostAssignment(pts0,pts1, max_children=2):
  pts0,pts1 = array(pts0),array(pts1)
  N = pts0.shape[0]
  M = pts1.shape[0]
  pts0 = pts0.astype(np.float32)
  pts1 = pts1.astype(np.float32)

  # define costs
  costs = np.zeros((N,M),np.float32)
  for j in range(N):
    for k in range(M):
      costs[j,k] = np.linalg.norm(pts0[j]-pts1[k], ord=2) # euclidean
  costs0 = costs.copy()

  # determine edges by greedily minimizing costs
  edges = np.zeros((N,M),np.uint8)
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

  assert all(edges.sum(0)==1)

  return edges, costs0

def testMinCostTrackWithDivisions(N=10, D=1, b=2):
  # ltps = [rand(randint(50,70)) for _ in range(10)]
  # print(trackAndLabel(ltps))
  pts0  = rand(N,D) * b
  pts1  = pts0 + rand(N,D)
  pts2  = cat( [pts0 + rand(N,D) , rand(N//5,D)*b], axis=0)
  edges, costs0 = greedyMinCostAssignment(pts0,pts1, max_children=2)
  plt.figure()
  plt.imshow(edges)
  plt.figure()
  plt.imshow(costs0)




