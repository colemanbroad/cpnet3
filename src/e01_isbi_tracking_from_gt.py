from tracking2 import *
from time import time
from scipy.ndimage import zoom
from scipy.interpolate import RegularGridInterpolator as RGI
from matplotlib import pyplot as plt
import pandas as pd

"""
Wed Mar 29, 2023

This script performs experiments on Ground Truth detections without using any
detection model. It explores solely the properties of linkers. The
experiments are designed to simulate real detections of varying quality to
determine which linker properties are important in different datasets and
with different detection models.

The Experiments
===============

1. How well does nearest neighbour (NN) tracking do on perfect detections across
   all datasets? 
  
   We see that NN tracking results in excellent performance in all but a small
   number of datasets when the input is perfect, ground truth detections, but 
   taking voxel anisotropy into account is important for 3D data.

2. What are the exceptions to this story?

   The MSC, MuSC, H157 and MDA231 datasets have [problems] which make NN linkers
   less effective.

3. What if our detections are imperfect? For example what if there are false
   positives and false negatives?


The linking methods include :
- Nearest neighbour links without constraints
- greedy nearest neib with division constraint
- global nearest neib 
- Tracking by Assignment

Optional lineage tree post-processing / denoising :
- gap joining
- stub branch pruning
- division flickering filter

The evaluation metrics :
- centerpoint matching with symmetric nearest neighbours
- precision, recall and f1 scores for both detections and links
- specialized division scores which allow for flexibility in precise div timing
"""

# Introduce a small number of False Positive detections to the tracking results
# to simulate noise in the detection procedure.
def addFalsePositivesToDTPS(dtps, fp_rate=0.05):
  # first let's add false positives that are evenly distributed throughout the 
  # volume of existing detections 
  for t,pts in dtps.items():
    nd = len(pts[0])
    pts = array(pts, dtype=np.float)
    mi = pts.min(axis=0,)
    ma = pts.max(axis=0,)
    npts = ceil(len(pts)*fp_rate)
    newpts = np.random.rand(npts,nd)*(ma-mi) + mi
    dtps[t] = np.concatenate([pts, newpts], axis=0)

# Introduce a small number of False Negatives detections to the tracking results
# to simulate noise in the detection procedure.
def removeFalseNegativesFromDTPS(dtps, fn_rate=0.05):
  for t,pts in dtps.items():
    nd  = len(pts[0])
    pts = array(pts, dtype=np.float)
    # mi = pts.min(axis=0,)
    # ma = pts.max(axis=0,)
    # npts = ceil(len(pts)*fn_rate)
    mask = np.random.rand(len(pts)) < fn_rate
    newpts = pts[~mask]
    # ipdb.set_trace()
    dtps[t] = newpts


# Shift the positions of existing detections by adding random jitter, and also
# adding a spatially correlated drift. The correlated drift can by const across
# the entire image, or only locally correlated. The decay length of the spatial
# correlations can be controlled... In addition to spatial correlations in the 
# drift we could also add temporal correlations? This is similar to tissue flows
def shiftDetectionsInDTPS(dtps, scale=0.05):
  # First we add random jitter
  for t,pts in dtps.items():
    nd  = len(pts[0])
    pts = array(pts, dtype=np.float)
    mi = pts.min(axis=0,)
    ma = pts.max(axis=0,)

    t1 = (4,)*nd + (nd,)
    t2 = (8,)*nd + (1,)
    drift = np.random.randn(*t1) * (ma-mi)*scale
    drift = zoom(drift, t2, order=3)
    # Plots look good
    # plt.quiver(drift[:,:,0], drift[:,:,1])
    # Now we eval drift at each p in pts.
    grid = tuple((np.linspace(mi[i],ma[i],n) for i,n in enumerate(drift.shape[:-1])))
    # linear interpolation between `drift` regions
    rgi  = RGI(grid,drift,method='linear')
    dtps[t] = pts + rgi(pts) # add correlated drift


# directory: path of e.g. 01_GT/TRA/
# method in ['nn', 'nn-prune', 'greedy', 'munkres']
def computeScores(directory, method):
  isbiname, dataset = path.normpath(directory).split(path.sep)[-3:-1]
  isbi = load_isbi_csv(isbiname)
  gt = loadISBITrackingFromDisk(directory)
  dtps = {t:[gt.pts[(t,n)] for n in gt.time2labelset[t]] for t in gt.times}
  # aniso = [1,1] if '2D' in directory else [1,1,1]
  aniso = isbi['voxelsize']
  dub = 100

  # addFalsePositivesToDTPS(dtps,fp_rate=0.10)
  # removeFalseNegativesFromDTPS(dtps, fn_rate=0.10)
  # shiftDetectionsInDTPS(dtps, scale=0.05)

  tic = time()
  if method=='nn':
    yp = link_nearestNeib(dtps=dtps, aniso=aniso, dub=dub)
  elif method=='nn-prune':
    yp = link_nearestNeib(dtps=dtps, aniso=aniso, dub=dub)
    addIsbiLabels(yp)
    pruneSingletonBranches(yp)
  elif method=='greedy':
    yp = link_minCostAssign(dtps=dtps, aniso=aniso, dub=dub, greedy=True)
  elif method=='munkres':
    yp = link_minCostAssign(dtps=dtps, aniso=aniso, dub=dub, greedy=False)
  delta_t = time() - tic

  scores = compare_trackings(gt,yp,aniso,dub)
  scores['delta_t']  = delta_t
  scores['isbiname'] = isbiname
  scores['dataset']  = dataset
  return scores

# matplot plot for F1 linking scores over time
def plotscores_bytime(scores):

  times = list(scores.keys())

  # vals = [s.n_gt for s in scores.values()]
  # plt.plot(times,vals,'o', label='n_gt')
  # vals = [s.n_proposed for s in scores.values()]
  # plt.plot(times,vals,'.', label='n_proposed')
  # vals = [s.n_matched for s in scores.values()]
  # plt.plot(times,vals,'.', label='n_matched')
  # plt.legend()

  plt.figure()
  vals = [s.f1 for s in scores.values()]
  plt.plot(times,vals,'.', label='F1')
  plt.legend()



# Only run once per dataset, then add it to isbidata.csv
def computeEnterExitCounts(directory):
  isbiname, dataset = path.normpath(directory).split(path.sep)[-3:-1]
  # isbi = load_isbi_csv(isbiname)

  # print(directory)
  d = f'../cpnet-out/{isbiname}/{dataset}/track-analysis/'
  # if path.exists(d + 'gt-tracks.pkl'): return
  # os.makedirs(d,exist_ok=True)
  # gt = loadISBITrackingFromDisk(directory)
  # pickle.dump(gt,open(d + 'gt-tracks.pkl','wb'))
  # return
  gt = pickle.load(open(d + 'gt-tracks.pkl','rb'))

  #   return locals()
  # def next(D):
  #   globals().update(D)
  
  # dtps = {t:[gt.pts[(t,n)] for n in gt.time2labelset[t]] for t in gt.times}
  # ipdb.set_trace()

  c = SN(entries=0, exits=0, total=0, division=0)
  t_start, t_final = min(gt.times), max(gt.times)

  for t,ls in gt.time2labelset.items():
    for l in ls:
      c.total += 1
      if gt.parents[(t,l)] is None and t!=t_start:
        c.entries += 1
      x = gt.children.get((t,l), None)
      if x is None and t!=t_final:
        c.exits += 1
      if x is not None and len(x)==2:
        c.division += 1

  print(directory, c)
  c.isbiname = isbiname
  c.dataset = dataset
  c.ratio = (c.entries + c.exits ) / c.total
  return c.__dict__

# pretty print nested python structures
def printscores(scores):
  print()
  # key = list(scores.keys())[0]
  # print(key)
  def printkv(k,v):
    if type(v) is float:
      print(f"{k:20s}{v:.4f}")
    else:
      print(f"{k:20s}{v}")
  print(scores['isbiname'] , scores['dataset'])
  for k,v in scores[key]['node'].items(): printkv(k,v)
  for k,v in scores[key]['edge'].items(): printkv(k,v)
  printkv("delta_t", scores[key]['delta_t'])
  print(scores[key]['node_confusion'])
  print(scores[key]['edge_confusion'])
  print()


# TODO
# directory is e.g. 'path/isbiname/01_GT/TRA/'
def parseDirectoryName(directory):
  isbiname, dataset = path.normpath(directory).split(path.sep)[-3:-1]


# entrypoint to build a table of scores over all GT directories
def scoreAllDirs():
  table = list()
  # for dire in glob("../data-raw/*/*_GT/TRA/"):

  base = path.normpath("../cpnet-out/")
  # gt_directories = "/projects/project-broaddus/rawdata/isbi_train/*/*_GT/TRA/"
  gt_directories = "../data-raw/*/*_GT/TRA/"
  res = []
  for directory in sorted(glob(gt_directories)):
    # print(directory)
    x = computeEnterExitCounts(directory)
    res.append(x)
    continue
    isbiname, dataset = path.normpath(directory).split(path.sep)[-3:-1]
    # if 'Fluo-C3DH-A549' not in isbiname: continue
    # if 'PhC-C2DL-PSC' in isbiname: continue
    if isbiname not in ['DIC-C2DH-HeLa', 'Fluo-C2DL-MSC', 'Fluo-C3DL-MDA231']: continue
    if dataset!='02_GT': continue
    outdir = path.join(base, isbiname, dataset, 'track-analysis')
    os.makedirs(outdir, exist_ok=True)

    # try:
    #   assert False
    #   d = pickle.load(open(outdir + '/compare_results-aniso-mca.pkl','rb'))
    #   if len(d)==0:
    #     print("Empty Data!")
    #     assert False
    #   # pickle.dump(d, open(outdir + '/compare_results-aniso-mca.pkl','wb'))
    #   # os.remove(directory + '/compare_results-aniso-mca.pkl')
    # except:

    for method in ['nn','nn-prune','greedy','munkres']:
      scores = computeScores(directory, method)
      scores['method'] = method
      pickle.dump(scores, open(outdir + f'/scores-{method}.pkl','wb'))
      table.append(scores)

    # printscores(scores)

  pd.DataFrame(res).to_csv('../isbi-tracking-stats-gt.csv')

  lines = []
  for i, scores in enumerate(table):
    node = scores['node']
    edge = scores['edge']
    scores.update({'node-'+k2:v2 for k2,v2 in node.items()})
    scores.update({'edge-'+k2:v2 for k2,v2 in edge.items()})
    del scores['node']
    del scores['edge']
    del scores['node_confusion']
    del scores['edge_confusion']
    lines.append(scores)

  print(tabulate(lines, headers='keys'))
  return lines

def saveTrackingImages(directory):
  directory = "Fluo-C2DL-MSC/"






if __name__=='__main__':
  res = scoreAllDirs()
  scores = formatTable(res)