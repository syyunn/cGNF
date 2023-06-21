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

import argparse
datasets = ["s1", "s2", "s3", "new_dataset"]

# Define argument parser with default values
parser = argparse.ArgumentParser(description='')
parser.add_argument("-load_config", default=None, type=str)

# General Parameters
parser.add_argument("-dataset", default='s1', choices=datasets, help="Which toy problem ?")
parser.add_argument("-load", default=False, action="store_true", help="Load a model ?")
parser.add_argument("-folder", default="", help="Folder")
parser.add_argument("-f_number", default=None, type=int, help="Number of heating steps.")
parser.add_argument("-test", default=False, action="store_true") # no model training. Set only testing by loading mdel and the dataset.
parser.add_argument("-nb_flow", type=int, default=1, help="Number of steps in the flow.")

# Optim Parameters
parser.add_argument("-weight_decay", default=1e-5, type=float, help="Weight decay value")
parser.add_argument("-learning_rate", default=3e-4, type=float, help="Weight decay value")
parser.add_argument("-nb_epoch", default=50000, type=int, help="Number of epochs")
parser.add_argument("-b_size", default=1024, type=int, help="Batch size")
# parser.add_argument("-b_size", default=128, type=int, help="Batch size")
parser.add_argument("-seed", default=31415, type=int, help="seed")

# Conditioner Parameters
parser.add_argument("-conditioner", default='DAG', choices=['DAG', 'Coupling', 'Autoregressive'], type=str)
parser.add_argument("-emb_net", default=[40, 30, 20], nargs="+", type=int, help="NN layers of embedding")
parser.add_argument("-nb_steps_dual", default=50, type=int, help="number of step between updating Acyclicity constraint and sparsity constraint")
parser.add_argument("-l1", default=0.5, type=float, help="Maximum weight for l1 regularization")
parser.add_argument("-gumble_T", default=0.5, type=float, help="Temperature of the gumble distribution.")

# Normalizer Parameters
parser.add_argument("-normalizer", default='monotonic', choices=['affine', 'monotonic'], type=str)
parser.add_argument("-int_net", default=[15, 10, 5], nargs="+", type=int, help="NN hidden layers of UMNN")
parser.add_argument("-nb_steps", default=50, type=int, help="Number of integration steps.")
parser.add_argument("-nb_estop", default=50, type=int, help="Number of epochs for early stopping.")
parser.add_argument("-n_mce_samples", default=2000, type=int, help="Number of Monte-Carlo mean estimation samples.")
parser.add_argument("-mce_b_size", default=2000, type=int, help="Monte-Carlo mean estimation Batch size")
parser.add_argument("-solver", default="CC", type=str, help="Which integral solver to use.",
                    choices=["CC", "CCParallel"])

# Print all the training parameters beings used for the current simulation
try:
    sys.argv = ['']
    args = parser.parse_args()
finally:
    sys.argv = temp_argv
print(sys.argv)
    
    
# create a folder to save all the logs/models for the current simulation
now = datetime.now()
dir_name = args.dataset if args.load_config is None else args.load_config
path = "Wodtke_sim_exp_logs/" + dir_name + "/" + now.strftime("%Y_%m_%d_%H_%M_%S") + '_' + socket.gethostname() if args.folder == "" else args.folder

if not(os.path.isdir(path)):
    os.makedirs(path)

print("dataset", args.dataset)    
# Start training the model by calling the training function from above    

from utils import train 

if __name__ == '__main__':

    model, data = train(args.dataset, 
                        load=args.load, 
                        path=path, 
                        nb_step_dual=args.nb_steps_dual, 
                        l1=args.l1, 
                        nb_epoch=args.nb_epoch, 
                        nb_estop=args.nb_estop,
                        int_net=args.int_net, 
                        emb_net=args.emb_net, 
                        b_size=args.b_size, 
                        all_args=args,
                        nb_steps=args.nb_steps, 
                        file_number=args.f_number,  
                        solver=args.solver, nb_flow=args.nb_flow,
                        train=not args.test, 
                        weight_decay=args.weight_decay, 
                        learning_rate=args.learning_rate,
                        cond_type=args.conditioner,  
                        norm_type=args.normalizer,  
                        n_mce_samples=args.n_mce_samples, 
                        mce_b_size=args.mce_b_size, 
                        seed=args.seed)

    x_max,_ = torch.max(data.trn.x, dim=0)
    x_min,_ = torch.min(data.trn.x, dim=0)

    print(data.trn.x[:2,:])
    print(data.trn.x.mean(0))
    print(data.trn.x.std(0))
    print(data.mu)
    print(data.sig)
    print(f'X_min = {x_min.numpy()}')
    print(f'X_max = {x_max.numpy()}')
    print(f'X_mu = {data.mu}')
    print(f'X_sigma = {data.sig}')


    from IPython.display import display
    from PIL import Image

    dag_path = './UCIdatasets/data/wodtke_sim/DAG.png'
    display(Image.open(dag_path))

    df_ds1 = data.df_ds1

    df_ds1.describe()

    df_ds1.head()

    df_ds1.describe()

    sns.pairplot(df_ds1)

