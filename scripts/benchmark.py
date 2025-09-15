
import sys
import os
import os.path
import numpy as np
import pandas as pd
import torch
import pickle
import subprocess
#from Bio import SeqIO
from importlib import reload

sys.path.append("/home/dahala/mnt/VAEproject/VAE-enzymes/vae_notebooks/notebooks")
sys.path.append("/home/dahala/mnt/VAEproject/VAE-enzymes/vae_notebooks/")

#os.chdir("/Users/dahala/GitHub/VAE-enzymes/vae_notebooks/notebooks/")

import minimal_version.parser_handler
#os.chdir("/Users/dahala/GitHub/VAE-enzymes/vae_notebooks/")

from minimal_version.preprocess_msa import Preprocessor, weight_sequences
from minimal_version.msa import MSA
from minimal_version.utils import Capturing, store_to_pkl, store_to_fasta, load_from_pkl

from minimal_version.train import setup_train, Train
from notebooks.minimal_version.benchmark import Benchmarker
os.chdir("/home/dahala/mnt/VAEproject/VAE-enzymes/vae_notebooks/notebooks")


CONFIGURATION_FILE = "msaEVC_GmSuSy_b05.json"
#CONFIGURATION_FILE = "msaEnzymeMiner_PtUGT1.json"
#CONFIGURATION_FILE = "msaGASP_bigMSA.json"
run = minimal_version.parser_handler.RunSetup(CONFIGURATION_FILE)
print(f" Working with {CONFIGURATION_FILE} configuration file!")
PFAM_INPUT = False

""" 
Benchmark model 
How well our model can regenarate sequences in various sets (positive, negative and training)
"""
N_SAMPLES = 500

Benchmarker(run, samples=N_SAMPLES, use_cuda=torch.cuda.is_available()).bench_dataset()
