import ipdb
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
plt.ion()
from glob import glob
import re
import pickle
from numpy import array, exp, zeros, maximum, indices
def ceil(x): return np.ceil(x).astype(int)
def floor(x): return np.floor(x).astype(int)

import json
from tabulate import tabulate

import sys
import shutil
import pandas

from pathlib import Path

import seaborn as sns
sns.set(style='ticks')

isbinames = ["Fluo-C2DL-Huh7" ,
  "DIC-C2DH-HeLa" ,
  "PhC-C2DH-U373" ,
  "Fluo-N2DH-GOWT1" ,
  "Fluo-C2DL-MSC" ,
  "Fluo-C3DL-MDA231" ,
  "Fluo-N2DH-SIM+" ,
  "PhC-C2DL-PSC" ,
  "Fluo-N2DL-HeLa" ,
  "Fluo-N3DH-CHO" ,
  "Fluo-C3DH-A549" ,
  "Fluo-C3DH-A549-SIM" ,
  "BF-C2DL-MuSC" ,
  "BF-C2DL-HSC" ,
  "Fluo-N3DH-SIM+" ,
  "Fluo-N3DH-CE" ,
  "Fluo-C3DH-H157" ,
  "Fluo-N3DL-DRO" ,
  "Fluo-N3DL-TRIC" ,
  "flywing",
  ]

metriclist = ['f1', 'loss', 'height']

def load_pkl(name): 
  with open(name,'rb') as file:
    return pickle.load(file)

def wipedir(path):
  path = Path(path)
  if path.exists(): shutil.rmtree(path)
  path.mkdir(parents=True, exist_ok=True)



## timeseries of all validation metrics for single dataset.
def plotHistory(isbiname):

  # PR = params()

  try:
    # history = load_pkl(f"/Users/broaddus/Desktop/mpi-remote/project-broaddus/cpnet3/cpnet-out/{isbiname}/train/history.pkl")
    # history = load_pkl(PR.savedir/"train/history.pkl")
    history = load_pkl(f"/Users/broaddus/work/isbi/cpnet3/cpnet-out/{isbiname}/train/history.pkl")
  except:
    return

  print(history)

  fig, ax = plt.subplots(nrows=4,sharex=True, )

  ax[0].plot(np.log(history.lossmeans), label="log train loss")
  ax[0].legend()

  valis = np.array(history.valimeans)

  ax[0+1].plot(np.log(valis[:,0]), label="log vali loss")
  ax[0+1].legend()

  ax[1+1].plot(valis[:,1], label="f1")
  ax[1+1].legend()

  ax[2+1].plot(valis[:,2], label="height")
  ax[2+1].legend()

  # plt.show()
  # y = input("'y' to save: ")
  # if y=='y': 
  plt.savefig(f"plots/history_{isbiname}.pdf")
  plt.close()

## timeseries of single metric for all datasets.
def plotAllHistories(metric = 'f1'):

  assert metric in metriclist

  # isbinames = [re.search(r"cpnet-out/(.*)/train/", x).group(1) for x in allhistories]
  # isbinames = isbidata.isbi_by_size
  # allhistories = sorted(glob("/Users/broaddus/Desktop/mpi-remote/project-broaddus/cpnet3/cpnet-out/*/train/history.pkl"))
  allhistories = [f"/Users/broaddus/Desktop/mpi-remote/project-broaddus/cpnet3/cpnet-out/{name}/train/history.pkl" for name in isbinames]

  fig, ax = plt.subplots(nrows=len(isbinames),sharex=True, sharey=True)

  for i,name in enumerate(allhistories):
    if not Path(name).exists(): continue
    history = load_pkl(name)
    vali = array(history.valimeans)
    # if isbinames[i]=='Fluo-N3DH-CHO': ipdb.set_trace()

    # Shrink current axis by 20%
    box = ax[i].get_position()
    ax[i].set_position([box.x0, box.y0, box.width * 0.6, box.height])

    if metric=='loss':
      ax[i].plot(np.log10(vali[:,0]), label=f"{isbinames[i]}") ## vali loss
    if metric=='f1':
      ax[i].plot(vali[:,1], label=f"{isbinames[i]}") ## f1 detection
    if metric=='height':
      ax[i].plot(vali[:,2], label=f"{isbinames[i]}") ## max height of output

    # ax[i].plot(np.log(history.lossmeans), label=f"{isbinames[i]}")
    ax[i].legend(loc='center left', bbox_to_anchor=(1, 0.5))

  plt.suptitle(f"{metric} Detection metric")
  plt.gcf().set_size_inches(6.4,9.46)
  # plt.tight_layout()
  # plt.show()
  plt.savefig(f"plots/allHistories_{metric}.pdf")
  plt.close()

## scatterplot of f1 scores
def plotMetrics():

  dirs = r"/Users/broaddus/Desktop/mpi-remote/project-broaddus/cpnet3/cpnet-out/(?P<isbiname>[^/]+)/predict/scores/matching.pkl"
  matchings = glob(dirs.replace(r'(?P<isbiname>[^/]+)','*'))
  alltables = []
  for m in matchings:
    isbiname = re.fullmatch(dirs, m).group('isbiname')
    matching_table = load_pkl(m)
    for row_dict in matching_table: row_dict['isbi'] = isbiname
    alltables += matching_table
    print(isbiname)

  alltables = pandas.DataFrame(alltables)
  pr = alltables.groupby(['isbi','dset'])[['precision','recall']].mean().sort_values('recall')

  print(pr)

  isbi_sorted = alltables.groupby('isbi')[['recall']].mean().sort_values('recall')
  # alltables = alltables.reindex(pr.index)
  # print(isbi_sorted)

  # sys.exit(0)
  # ipdb.set_trace()

  # print(alltables)
  facetgrid = sns.relplot(data=alltables, x='precision',y='recall',hue='dset',col='isbi',col_wrap=6, col_order=isbi_sorted.index, height=1.5, aspect=1)
  # df[['POINTS','PRICE','short-name']].apply(lambda row: facetgrid.ax.text(*row),axis=1);
  plt.gcf().set_size_inches(14.12,  9.1)
  # plt.show()
  # input()
  # ipdb.set_trace()
  savedir = r"/Users/broaddus/work/isbi/cpnet3/results/plots/"
  plt.savefig(savedir + "precision-recall-newie.pdf")
  # plt.savefig("../results/plots/precision-recall.pdf")



badkeys = ['gt_matched_mask', 'yp_matched_mask', 'gt2yp', 'yp2gt', 'pts_gt', 'pts_yp']
keys = ['isbi', 'time', 'weights', 'n_matched', 'n_proposed', 'n_gt', 'precision', 'recall', 'f1']

def printdikt(dikt):
  for k,v in dikt.items():
    print(k)
    for k2,v2 in v.__dict__.items():
      if k2 not in keys: continue
      print("  ",k2,v2)

if __name__=='__main__':

  if 'wipe' == sys.argv[1]:
    wipedir("plots")
  elif 'pr' == sys.argv[1]:
    plotMetrics()
  elif 'all' == sys.argv[1]:
    for met in metriclist:
      plotAllHistories(met)
  elif 'one' == sys.argv[1]:
    for isbi in isbinames:
      plotHistory(isbi)
  if sys.argv[1] in isbinames:
    plotHistory(sys.argv[1])
  else:
    print("Error: Args don't match any task.")
