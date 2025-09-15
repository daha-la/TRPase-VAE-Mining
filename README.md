# TRPase VAE Mining and Visualization
This repository contains all data and code to replicate the Variational Autoencoder (VAE) mining study for visualizing the sequence landscape of TRPases and identifying novel candidates as described in de Boer et al. 

This study was based on the work done by Kohout et al., with all VAE training following the specifics of their original pipeline. 

To run the code, please create a conda environment using the requirement file:
```bash
conda create -n vae_env python=3.10
conda activate vae_env
conda install pytorch torchvision -c pytorch
pip install -r requirements.txt
```

## Alignments
Contains the alignment used to create the Hidden Markov Model (HMM)

## Extra Data
Contains all extra data not used directly in VAE training

## HMM
Contains the HMM used to realign all sequences to create the VAE training MSA

## VAE Notebooks
Contains the data and code used to in the VAE workflow, with "datasets" containing the training MSA, "notebooks" containing the code used for both training and visualizing the VAE latent space, and "results" containing all VAE results and figures.
