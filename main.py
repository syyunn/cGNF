import datetime
now = datetime.datetime.now()
print("Code run on date and time for reproducibility of results and plots : ")
print(now.strftime("%Y-%m-%d %H:%M:%S"))

# General helper packages
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'#0,1,2,3
import socket
import numpy as np
np.set_printoptions(precision=3, suppress=None)
import datetime
import math
import random
import sys
temp_argv = sys.argv
import timeit
from timeit import default_timer as timer
import lib.utils as utils
from datetime import datetime
import re
import yaml

# packages for data handling and plotting/printing
import pandas as pd
import seaborn as sns
import graphviz
import matplotlib
from matplotlib import cm

# %matplotlib notebook
# %matplotlib inline  

from matplotlib import pyplot as plt
from dowhy import CausalModel
import networkx as nx
from PIL import Image
from tqdm import tqdm
plt.rcParams['figure.dpi'] = 100 # https://blakeaw.github.io/2020-05-25-improve-matplotlib-notebook-inline-res/
plt.rcParams['savefig.dpi'] = 300
sns.set(rc={"figure.dpi":100, 'savefig.dpi':300})
sns.set_context('notebook')
sns.set_style("ticks")

# Pytorch
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
import torch.cuda
import torch.backends.cudnn as cudnn
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal

# Import dataset handling (preprocessing+saving+loading) module
import UCIdatasets

# Import modules from GNF for the monotonic normalizer/transformer and graphical conditioner
from models.Normalizers import *
from models.Conditionners import *
from models.NormalizingFlowFactories import buildFCNormalizingFlow
from models.NormalizingFlow import *

cond_types = {"DAG": DAGConditioner, "Coupling": CouplingConditioner, "Autoregressive": AutoregressiveConditioner} # types of conditioners
norm_types = {"affine": AffineNormalizer, "monotonic": MonotonicNormalizer} # types of transformers/normalizers
