import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gc
import sys
import psutil
import os


def receptivefield(net,kern=(3,5,5)):
  """
  calculate receptive field / receptive kernel of Conv net
  WARNING: overwrites net weights. save weights first!
  WARNING: only works with relu activations, pooling and upsampling.
  """
  def rfweights(m):
    if type(m) in [nn.Conv3d,nn.Conv2d]:
      m.weight.data.fill_(1/np.prod(kern)) ## conv kernel 3*5*5
      m.bias.data.fill_(0.0)
  net.apply(rfweights);
  if len(kern)==3:
    x0 = np.zeros((256,256,256)); x0[128,128,128]=1;
  elif len(kern)==2:
    x0 = np.zeros((512,512)); x0[256,256]=1;
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  xout = net.to(device)(torch.from_numpy(x0)[None,None].float().to(device)).detach().cpu().numpy()
  return xout

def init_weights(net):
  def f(m):
    if type(m) == nn.Conv3d:
      torch.nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
      m.bias.data.fill_(0.05)
  net.apply(f);

def test_weights(net):
  table = ["weight mean,weight std,bias mean,bias std".split(',')]
  def f(m):
    if type(m) == nn.Conv3d:
      table.append([float(m.weight.data.mean()),float(m.weight.data.std()),float(m.bias.data.mean()),float(m.bias.data.std())])
      print(m)
  net.apply(f)
  return table


def n_conv(chans, kernsize, padding):
  conv = {2:nn.Conv2d,3:nn.Conv3d}[len(kernsize)]
  res = []
  for i in range(len(chans)-1):
    res.append(conv(chans[i],chans[i+1],kernsize, padding=padding))
    res.append(nn.ReLU())
  res = nn.Sequential(*res)
  return res

class Unet3(nn.Module):
  """
  Unet with 3 pooling steps.
  """

  def __init__(self, c=32, io=[[1],[1]], finallayer=nn.LeakyReLU, pool=(1,2,2), kernsize=(3,5,5)):
    super(Unet3, self).__init__()

    self.pool = pool
    self.kernsize = kernsize
    self.finallayer = finallayer
    pad = tuple(x//2 for x in kernsize)

    self.l_ab = n_conv([io[0][0], c, c,], kernsize, padding=pad)
    self.l_cd = n_conv([1*c, 2*c,  2*c,], kernsize, padding=pad)
    self.l_ef = n_conv([2*c, 4*c,  4*c,], kernsize, padding=pad)
    self.l_gh = n_conv([4*c, 8*c,  4*c,], kernsize, padding=pad)
    self.l_ij = n_conv([8*c, 4*c,  2*c,], kernsize, padding=pad)
    self.l_kl = n_conv([4*c, 2*c,  1*c,], kernsize, padding=pad)
    self.l_mn = n_conv([2*c, 1*c,  1*c,], kernsize, padding=pad)

    conv = {2:nn.Conv2d,3:nn.Conv3d}[len(kernsize)]
    self.l_o  = nn.Sequential(conv(1*c,io[1][0],(1,)*len(kernsize),padding=0), finallayer())

  def forward(self, x):

    maxpool = {2:nn.MaxPool2d,3:nn.MaxPool3d}[len(self.kernsize)]

    c1 = self.l_ab(x)
    c2 = maxpool(self.pool)(c1)
    c2 = self.l_cd(c2)
    c3 = maxpool(self.pool)(c2)
    c3 = self.l_ef(c3)
    c4 = maxpool(self.pool)(c3)
    c4 = self.l_gh(c4)
    c4 = F.interpolate(c4,scale_factor=self.pool)
    c4 = torch.cat([c4,c3],1)
    c4 = self.l_ij(c4)
    c4 = F.interpolate(c4,scale_factor=self.pool)
    c4 = torch.cat([c4,c2],1)
    c4 = self.l_kl(c4)
    c4 = F.interpolate(c4,scale_factor=self.pool)
    c4 = torch.cat([c4,c1],1)
    c4 = self.l_mn(c4)
    out1 = self.l_o(c4)

    return out1

class Unet2(nn.Module):
  """
  The same exact shape we used for models submitted to ISBI. 2,767,777 params.
  """

  def __init__(self, c=32, io=[[1],[1]], finallayer=nn.LeakyReLU, pool=(1,2,2), kernsize=(3,5,5)):
    super(Unet2, self).__init__()

    self.pool = pool
    self.kernsize = kernsize
    self.finallayer = finallayer
    pad = tuple(x//2 for x in kernsize)

    self.l_ab = n_conv([io[0][0], c, c,], kernsize, padding=pad)
    self.l_cd = n_conv([1*c, 2*c,  2*c,], kernsize, padding=pad)
    self.l_ef = n_conv([2*c, 4*c,  2*c,], kernsize, padding=pad)
    self.l_gh = n_conv([4*c, 2*c,  1*c,], kernsize, padding=pad)
    self.l_ij = n_conv([2*c, 1*c,  1*c,], kernsize, padding=pad)

    conv = {2:nn.Conv2d,3:nn.Conv3d}[len(kernsize)]
    self.l_o  = nn.Sequential(conv(1*c,io[1][0],(1,)*len(kernsize),padding=0), finallayer())

  def forward(self, x):

    maxpool = {2:nn.MaxPool2d,3:nn.MaxPool3d}[len(self.kernsize)]

    c1 = self.l_ab(x)
    c2 = maxpool(self.pool)(c1)
    c2 = self.l_cd(c2)
    c3 = maxpool(self.pool)(c2)
    c3 = self.l_ef(c3)
    c3 = F.interpolate(c3,scale_factor=self.pool)
    c3 = torch.cat([c3,c2],1) # concat on channels
    c3 = self.l_gh(c3)
    c3 = F.interpolate(c3,scale_factor=self.pool)
    c3 = torch.cat([c3,c1],1)
    c3 = self.l_ij(c3)
    out1 = self.l_o(c3)

    return out1

class Unet1(nn.Module):
  """
  Small Unet for tiny data.
  """

  def __init__(self, c=32, io=[[1],[1]], finallayer=nn.LeakyReLU, pool=(1,2,2), kernsize=(3,5,5)):
    super(Unet1, self).__init__()

    self.pool = pool
    self.kernsize = kernsize
    self.finallayer = finallayer
    pad = tuple(x//2 for x in kernsize)

    self.l_ab = n_conv([io[0][0], c, c,], kernsize, padding=pad)
    self.l_cd = n_conv([1*c, 2*c,  1*c,], kernsize, padding=pad)
    self.l_ef = n_conv([2*c, 1*c,  1*c,], kernsize, padding=pad)

    conv = {2:nn.Conv2d,3:nn.Conv3d}[len(kernsize)]
    self.l_o  = nn.Sequential(conv(1*c,io[1][0],(1,)*len(kernsize),padding=0), finallayer())

  def forward(self, x):

    maxpool = {2:nn.MaxPool2d,3:nn.MaxPool3d}[len(self.kernsize)]

    c1 = self.l_ab(x)
    c2 = maxpool(self.pool)(c1)
    c2 = self.l_cd(c2)
    c3 = F.interpolate(c2,scale_factor=self.pool)
    c3 = torch.cat([c3,c1],1)
    c3 = self.l_ef(c3)
    out1 = self.l_o(c3)

    return out1

## resnets are incomplete

def conv_res(c0,c1,c2):
  return nn.Sequential(
    nn.Conv3d(c0,c1,(3,5,5),padding=(1,2,2)),
    # nn.BatchNorm3d(c2),
    nn.ReLU(),
    # nn.Dropout3d(p=0.1),
    nn.Conv3d(c1,c2,(3,5,5),padding=(1,2,2)),
    # nn.BatchNorm3d(c2),
    # nn.ReLU(),
    # nn.Dropout3d(p=0.1),
    )

class Res1(nn.Module):
  def __init__(self, c=32, io=[[1],[1]], finallayer=nn.LeakyReLU):
    super(Unet2, self).__init__()

    self.l_ab = conv2(io[0][0] ,1*c, 1*c)
    self.l_cd = conv2(1*c, 2*c, 2*c)
    self.l_ef = conv2(2*c, 1*c, 1*c)
    # self.l_gh = conv2(4*c, 2*c, 1*c)
    # self.l_ij = conv2(2*c, 1*c, 1*c)
    
    self.l_k  = nn.Sequential(nn.Conv3d(1*c,io[1][0],(1,1,1),padding=0), finallayer())

  def forward(self, x):

    c1 = nn.Relu()(self._lab(x)  + x)
    c2 = nn.Relu()(self._lcd(c1) + c1)
    c3 = nn.Relu()(self._lef(c2) + c2)
    out1 = self.l_k(c3)

    return out1


## utils

def pretty_size(size):
  """Pretty prints a torch.Size object"""
  assert(isinstance(size, torch.Size))
  return " × ".join(map(str, size))

def dump_tensors(gpu_only=True):
  """Prints a list of the Tensors being tracked by the garbage collector."""
  total_size = 0
  for obj in gc.get_objects():
    try:
      if torch.is_tensor(obj):
        if not gpu_only or obj.is_cuda:
          print("%s:%s%s %s" % (type(obj).__name__, 
                      " GPU" if obj.is_cuda else "",
                      " pinned" if obj.is_pinned else "",
                      pretty_size(obj.size())))
          total_size += obj.numel()
      elif hasattr(obj, "data") and torch.is_tensor(obj.data):
        if not gpu_only or obj.is_cuda:
          print("%s → %s:%s%s%s%s %s" % (type(obj).__name__, 
                           type(obj.data).__name__, 
                           " GPU" if obj.is_cuda else "",
                           " pinned" if obj.data.is_pinned else "",
                           " grad" if obj.requires_grad else "", 
                           " volatile" if obj.volatile else "",
                           pretty_size(obj.data.size())))
          total_size += obj.data.numel()
    except Exception as e:
      pass        
  print("Total size:", total_size)

def memReport():
  totalsize = 0
  for obj in gc.get_objects():
    if torch.is_tensor(obj):
      print(type(obj), obj.size(), obj.dtype)
      totalsize += obj.size().numel()

  print("Total Size: ", totalsize)
    
def cpuStats():
  print(sys.version)
  print(psutil.cpu_percent())
  print(psutil.virtual_memory())  # physical memory usage
  pid = os.getpid()
  proc = psutil.Process(pid)
  memoryUse = proc.memory_info()[0] / 2. ** 30  # memory use in GB...I think
  print('memory GB:', memoryUse)

