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

import isbidata
from pathlib import Path

def load_pkl(name): 
  with open(name,'rb') as file:
    return pickle.load(file)

## Plot all metrics for single dataset.
def plotHistory(isbiname):

  # PR = params()

  # allhistories = glob("/Users/broaddus/Desktop/mpi-remote/project-broaddus/cpnet3/cpnet-out/*/train/history.pkl")
  # isbinames = [re.match("cpnet-out/(.*)/train/", x).group(1) for x in allhistories]

  history = load_pkl(f"/Users/broaddus/Desktop/mpi-remote/project-broaddus/cpnet3/cpnet-out/{isbiname}/train/history.pkl")
  # history = load_pkl(PR.savedir/"train/history.pkl")
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

metriclist = ['f1', 'loss', 'height']

## Plot single metric across all datasets.
def plotAllHistories(metric = 'f1'):

  assert metric in metriclist

  # isbinames = [re.search(r"cpnet-out/(.*)/train/", x).group(1) for x in allhistories]
  isbinames = isbidata.isbi_by_size
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
  plt.tight_layout()
  plt.show()
  plt.gcf().set_size_inches(6.4,9.46)

  inp = input("Save? [y]: ")
  if inp in ["Y","y"]: 
    plt.savefig(f"plots/allHistories_{metric}.pdf")
    print(f"Figsize is {plt.gcf().get_size_inches()}")

import sys

if __name__=='__main__':
  for met in metriclist:
    plotAllHistories(met)
  # for isbi in isbidata.isbi_by_size:
  #   plotHistory(isbi)
