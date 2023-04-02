from tracking2 import *

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



def computeScores(directory):
  isbiname, dataset = path.normpath(directory).split(path.sep)[-3:-1]
  isbi = load_isbi_csv(isbiname)
  # directory = "data-isbi/DIC-C2DH-HeLa/01_GT/TRA/"
  gt = loadISBITrackingFromDisk(directory)
  dtps = {t:[gt.pts[(t,n)] for n in gt.time2labelset[t]] for t in gt.times}
  # aniso = [1,1] if '2D' in directory else [1,1,1]
  aniso = isbi['voxelsize']
  dub = 100
  yp = nn_tracking(dtps=dtps, aniso=aniso, dub=dub)
  pruneSingletonBranches(yp)
  # ipdb.set_trace()
  scores = compare_trackings(gt,yp,aniso,dub)
  return {(isbiname, dataset) : scores}

def scoreAllDirs():
  table = dict()
  # for dire in glob("../data-isbi/*/*_GT/TRA/"):

  base = path.normpath("../cpnet-out/")
  # pattern = "/projects/project-broaddus/rawdata/isbi_train/*/*_GT/TRA/"
  pattern = "../data-isbi/*/*_GT/TRA/"
  for directory in sorted(glob(pattern)):
    print(directory)
    isbiname, dataset = path.normpath(directory).split(path.sep)[-3:-1]
    outdir = path.join(base, isbiname, dataset, 'track-analysis')
    os.makedirs(outdir, exist_ok=True)
    # ipdb.set_trace()

    try:
      assert False
      d = pickle.load(open(outdir + '/compare_results-aniso.pkl','rb'))
      if len(d)==0:
        print("Empty Data!")
        assert False
      # pickle.dump(d, open(outdir + '/compare_results-aniso.pkl','wb'))
      # os.remove(directory + '/compare_results-aniso.pkl')
    except:
      d = computeScores(directory)
      pickle.dump(d, open(outdir + '/compare_results-aniso.pkl','wb'))

    table.update(d)
  return table

def saveTrackingImages(directory):
  directory = "Fluo-C2DL-MSC/"

def formatTable(res):
  lines = []
  for k,v in res.items():
    d = dict(name=k[0], dset=k[1][:2])
    d.update({'node-'+k2:v2 for k2,v2 in v.node.__dict__.items()})
    d.update({'edge-'+k2:v2 for k2,v2 in v.edge.__dict__.items()})
    lines.append(d)

  print(tabulate(lines, headers='keys'))
  return lines

if __name__=='__main__':
  res = scoreAllDirs()
  scores = formatTable(res)