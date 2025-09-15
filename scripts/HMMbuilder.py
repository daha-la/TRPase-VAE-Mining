
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


from minimal_version.preprocess_msa import Preprocessor, weight_sequences
from minimal_version.msa import MSA
from minimal_version.utils import Capturing, store_to_pkl, store_to_fasta, load_from_pkl

from minimal_version.train import setup_train, Train
os.chdir("/home/dahala/mnt/VAEproject/VAE-enzymes/vae_notebooks/notebooks")

folder_path = '/home/dahala/mnt/VAEproject/VAE-enzymes'

class HMMmodule:

    def __init__(self,run_config):

        self.muscle_path = run_config.muscle_path
        self.hmmer_path = run_config.hmmer_path

        self.msa_fasta = run_config.msa_fasta
        self.msa_name = run_config.msa_name
        self.hmm_model = run_config.hmm_model
        
    def MSA_create(self):
        print('Creating MSA using MUSCLE...')
        #subprocess.run(f'{self.muscle_path} -align {self.msa_fasta} -output {folder_path}/alignments/muscle_outputs/{self.msa_name}.afa > {folder_path}/alignments/muscle_outputs/muscle_{self.msa_name}.log', shell=True, executable="/bin/bash")
        print(f'{self.muscle_path} -in {self.msa_fasta} -out {folder_path}/alignments/muscle_outputs/{self.msa_name}.afa > {folder_path}/alignments/muscle_outputs/muscle_{self.msa_name}.log')
        subprocess.run(f'{self.muscle_path} -in {self.msa_fasta} -out {folder_path}/alignments/muscle_outputs/{self.msa_name}.afa > {folder_path}/alignments/muscle_outputs/muscle_{self.msa_name}.log', shell=True, executable="/bin/bash")
    
    def build(self):
        if not os.path.exists(f'{folder_path}/hmm_model/{self.hmm_model}.hmm'):
            print('Building HMM model...')
            print(f'{self.hmmer_path}/hmmbuild --amino {folder_path}/hmm_model/{self.hmm_model}.hmm {folder_path}/alignments/muscle_outputs/{self.msa_name}.afa > {folder_path}/hmm_model/{self.hmm_model}_hmmbuild.log')
            subprocess.run(f'{self.hmmer_path}/hmmbuild --amino {folder_path}/hmm_model/{self.hmm_model}.hmm {folder_path}/alignments/muscle_outputs/{self.msa_name}.afa > {folder_path}/hmm_model/{self.hmm_model}_hmmbuild.log', shell=True, executable="/bin/bash")


CONFIGURATION_FILE = "msaEVC_GmSuSy_b05.json"
run = minimal_version.parser_handler.RunSetup(CONFIGURATION_FILE)
print(f" Working with {CONFIGURATION_FILE} configuration file!")
PFAM_INPUT = False

HMMbuild = HMMmodule(run)
#HMMbuild.MSA_create()
HMMbuild.build()

