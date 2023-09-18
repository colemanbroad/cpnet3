
from tracking2 import *
from time import time
from scipy.ndimage import zoom
from scipy.interpolate import RegularGridInterpolator as RGI
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from skimage.io import imsave

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


# base = path.normpath("../cpnet-out/")
# cpnet_out = path.normpath("../cpnet-out/")
outdir = path.normpath("../results/e01/") + path.sep

# TODO
# directory is e.g. 'path/isbiname/01_GT/TRA/'
def parseDirectoryName(directory):
  isbiname, dataset = path.normpath(directory).split(path.sep)[-3:-1]



### FAULT INJECTION

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

### RUN EVALUATION, Make PLOTS and TABLES

# directory: path of e.g. 01_GT/TRA/
# method in ['nn', 'nn-prune', 'greedy', 'munkres']
def computeScores(directory, method):
  isbiname, dataset = path.normpath(directory).split(path.sep)[-3:-1]

  d = f'../cpnet-out/{isbiname}/{dataset}/track-analysis/'
  gt = pickle.load(open(d + 'gt-tracks.pkl','rb'))
  # gt = loadISBITrackingFromDisk(directory)

  dtps = {t:[gt.pts[(t,n)] for n in gt.time2labelset[t]] for t in gt.times}
  # aniso = [1,1] if '2D' in directory else [1,1,1]
  isbi = load_isbi_csv(isbiname)
  aniso = isbi['voxelsize']
  dub = 100

  # addFalsePositivesToDTPS(dtps,fp_rate=0.10)
  # removeFalseNegativesFromDTPS(dtps, fn_rate=0.10)
  # shiftDetectionsInDTPS(dtps, scale=0.05)

  best_hyperparams = []

  tic = time()
  if method=='nn':
    # ipdb.set_trace()
    best = train_link_nearestNeib(gt,aniso)
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

  ipdb.set_trace()

  img_shape = isbi['rawsize 01'] if dataset=='01_GT' else isbi['rawsize 02']
  g = drawTrackingTailsWithErrorsGenerator(gt, yp, scores['edge_matches'], img_shape)
  
  from cpnet import img2png, norm_minmax01

  os.makedirs(d + 'png/', exist_ok=True)
  for i,lab in enumerate(g):
    # if i < 1200: continue
    # if i == 1231: break
    raw = imread(f"../data-raw/{isbiname}/{dataset[:2]}/t{i:03d}.tif")
    # rawpng = img2png(raw, 'I')
    raw = norm_minmax01(raw)
    raw = img2png(raw, 'I')
    composite = np.round(raw/2 + lab/2).astype(np.uint8).clip(min=0,max=255)
    imsave(d + 'png/' + f'lab{i:04d}.png', composite)


  node = scores['node']
  edge = scores['edge']
  scores.update({'node-'+k:v for k,v in node.items()})
  scores.update({'edge-'+k:v for k,v in edge.items()})
  del scores['node']
  del scores['edge']
  del scores['node_confusion']
  del scores['edge_confusion']
  del scores['node_timeseries']
  del scores['edge_by_time']

  scores['delta_t']  = delta_t
  scores['isbiname'] = isbiname
  scores['dataset']  = dataset
  scores['method'] = method

  # ipdb.set_trace()
  return scores

# Compute Enter/Exit/Division statistics on ground truth ISBI trackings.
def computeEnterExitCounts(directory):
  isbiname, dataset = path.normpath(directory).split(path.sep)[-3:-1]

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

# Create isbi-tracking-stats-gt.csv
def csv_IsbiTrackingStatsGT():
  gt_directories = "../data-raw/*/*_GT/TRA/"
  
  table = []
  for directory in sorted(glob(gt_directories)):    
    print(directory)
    x = computeEnterExitCounts(directory)
    table.append(x)
  pd.DataFrame(table).to_csv(outdir + 'isbi-tracking-stats-gt.csv')

# entrypoint to build a table of scores over all GT directories
def csv_scoreAllDirs():
  gt_directories = "../data-raw/*/*_GT/TRA/"
  gt_directories = "../data-raw/Fluo-C2DL-MSC/02_GT/TRA/"
  # Fluo-C2DL-MSC
  table = list()
  for directory in sorted(glob(gt_directories)):
        
    isbiname, dataset = path.normpath(directory).split(path.sep)[-3:-1]

    for method in ['nn','nn-prune','greedy','munkres']:
      if (isbiname,dataset,method) in skip_experiments: continue
      print(directory, method)
      scores = computeScores(directory, method)
      table.append(scores)

  pd.DataFrame(table).to_csv(outdir + "scoreAllDirs2.csv")

  # print(tabulate(lines, headers='keys'))
  # return lines

# Each of these experiments take more than ten seconds
skip_experiments = [
  ("Fluo-N3DH-CE", "01_GT", "greedy"),
  ("PhC-C2DL-PSC", "02_GT", "munkres"),
  ("Fluo-N3DH-CE", "02_GT", "greedy"),
  ("PhC-C2DL-PSC", "01_GT", "munkres"),
  ("PhC-C2DL-PSC", "02_GT", "greedy"),
  ("PhC-C2DL-PSC", "01_GT", "greedy"),
  ]

def plot_scoreAllDirs():
  df = pd.read_csv("../scoreAllDirs.csv")

  df['log10 err-rate'] = np.log10(1 - df['edge-f1']) ## 1 - F1 propto Error Rate
  df['isbiname/dataset'] = df['isbiname'] + ' / ' + df['dataset']

  # sns.set(style='ticks')
  # fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True)
  # for idx, row in enumerate(df):
  #   ax.plot(row['edge-f1'], row['isbiname'], c=row['method'])

  facetgrid = sns.relplot(
    data=df,
    x='log10 err-rate',
    y='isbiname/dataset',
    hue='method',
    # col='isbi',
    # col_wrap=6, 
    # col_order=isbi_sorted.index, 
    height=1.5, 
    aspect=1,
    )
  plt.gcf().set_size_inches(14.12,  9.1)

  # plt.show()
  # input()
  # ipdb.set_trace()
  plt.savefig("../results/plots/plot_scoreAllDirs.pdf")
  plt.close()


# This function should extract <keyword> phrases from our regex string
# and 
def reglob(re_str):
  c = re.compile(re_str)


def getRawImageSizes():
  for ds in glob('../data-raw/*/*/t000*.tif'):
    m = re.search('data-raw/')
    if 't000.tif' not in ds and 't0000.tif' not in ds: continue
    print(ds, imread(ds).shape)


# WIP: helper func. matplot plot for F1 linking scores over time
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


### TRAIN/OPTIMIZE LINKING METHODS

def csv_hyperSearch_link_nearestNeib():
  gt_directories = "../data-raw/*/*_GT/TRA/"
  
  table = []
  for directory in sorted(glob(gt_directories)):
    print(directory)
    isbiname, dataset = path.normpath(directory).split(path.sep)[-3:-1]
    d = f'../cpnet-out/{isbiname}/{dataset}/track-analysis/'
    gt = pickle.load(open(d + 'gt-tracks.pkl','rb'))
    dtps = {t:[gt.pts[(t,n)] for n in gt.time2labelset[t]] for t in gt.times}
    # aniso = [1,1] if '2D' in directory else [1,1,1]
    isbi = load_isbi_csv(isbiname)
    aniso = isbi['voxelsize']
    best = train_link_nearestNeib(gt,aniso)
    best.isbiname = isbiname
    best.dataset = dataset
    table.append(best.__dict__)
    
  pd.DataFrame(table).to_csv(outdir + 'hyperSearch_link_nearestNeib.csv')


### Save tracking images for each 

def saveTrackingImages(directory):
  directory = "Fluo-C2DL-MSC/"


if __name__=='__main__':
  scoreAllDirs()



