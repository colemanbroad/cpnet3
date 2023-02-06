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


def load_pkl(name): 
  with open(name,'rb') as file:
    return pickle.load(file)

def plotHistory():

  # PR = params()

  # allhistories = glob("/Users/broaddus/Desktop/mpi-remote/project-broaddus/cpnet3/cpnet-out/*/train/history.pkl")
  # isbinames = [re.match("cpnet-out/(.*)/train/", x).group(1) for x in allhistories]

  history = load_pkl("/Users/broaddus/Desktop/mpi-remote/project-broaddus/cpnet3/cpnet-out/Fluo-N3DH-CHO/train/history.pkl")
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

def plotAllHistories():

  # PR = params()

  allhistories = glob("/Users/broaddus/Desktop/mpi-remote/project-broaddus/cpnet3/cpnet-out/*/train/history.pkl")
  # ipdb.set_trace()
  isbinames = [re.search(r"cpnet-out/(.*)/train/", x).group(1) for x in allhistories]
  fig, ax = plt.subplots(nrows=len(isbinames),sharex=True, sharey=True,)
  metric = 'f1' ## 'f1' ## 'loss' 'max height'

  for i,name in enumerate(allhistories):
    history = load_pkl(name)
    # Shrink current axis by 20%
    box = ax[i].get_position()
    ax[i].set_position([box.x0, box.y0, box.width * 0.6, box.height])
    if metric=='f1':
      ax[i].plot(array(history.valimeans)[:,1], label=f"{isbinames[i]}") ## f1 detection
    if metric=='loss':
      ax[i].plot(np.log10(array(history.valimeans)[:,0]), label=f"{isbinames[i]}") ## vali loss
    if metric=='max height':
      ax[i].plot(array(history.valimeans)[:,2], label=f"{isbinames[i]}") ## max height of output

    # ax[i].plot(np.log(history.lossmeans), label=f"{isbinames[i]}")
    ax[i].legend(loc='center left', bbox_to_anchor=(1, 0.5))

  plt.suptitle(f"{metric} Detection metric")
  plt.tight_layout()
  plt.show()

if __name__=='__main__':
  plotAllHistories()
  input()
  plt.savefig("plotAllHistories.pdf")