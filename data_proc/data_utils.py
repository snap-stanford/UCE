import warnings
warnings.filterwarnings("ignore")

import scanpy as sc
import torch

from torch import nn, Tensor
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import numpy as np
import pickle
import os
import argparse
import logging
import time

from tqdm.auto import tqdm
import pandas as pd

import math
import anndata
from pathlib import Path


from torch.utils.data import dataset
from torch.utils.data import DataLoader, TensorDataset, dataset
from scipy.stats import binom
from typing import Dict, List, Optional, Tuple
from scanpy import AnnData


from data_proc.gene_embeddings import load_gene_embeddings_adata

def data_to_torch_X(X):
    if isinstance(X, sc.AnnData):
        X = X.X
    if not isinstance(X, np.ndarray):
            X = X.toarray()
    return torch.from_numpy(X).float()

class SincleCellDataset(data.Dataset):
    def __init__(self,
                expression: torch.tensor, # Subset to hv genes, count data! cells x genes
                protein_embeddings: torch.tensor, # same order as expression, also subset genes x pe
                labels: None, # optional, tensor of labels
                covar_vals: None, # tensor of covar values or none
                ) -> None:
        super(SincleCellDataset, self).__init__()
        
        # Set expression
        self.expression = expression
        
        row_sums = self.expression.sum(1) # UMI Counts
        log_norm_count_adj = torch.log1p(self.expression / (self.expression.sum(1)).unsqueeze(1) * torch.tensor(1000))       
        
        # Set log norm and count adjusted expression
        max_vals, max_idx = torch.max(log_norm_count_adj, dim=0)
        self.expression_mod =  log_norm_count_adj / max_vals
        
        # Calculate dropout likliehoods of each gene
        self.dropout_vec = (self.expression == 0).float().mean(0) # per gene dropout percentages
        
        # Set data info
        self.num_cells = self.expression.shape[0]
        self.num_genes = self.expression.shape[1]
        
        # Set optional label info, including categorical covariate index
        self.covar_vals = covar_vals
        self.labels = labels
        
        # Set protein embeddings
        self.protein_embeddings = protein_embeddings
        
        self.item_mode = "expression"
        if self.covar_vals is not None:
            self.item_mode = "expression+covar"
        
        
    def __getitem__(self, idx):
        if self.item_mode == "expression":
            if isinstance(idx, int):
                if idx < self.num_cells:
                    return self.expression[idx, :]
                else:
                    raise IndexError
            else:
                raise NotImplementedError
        elif self.item_mode == "expression+covar":
            if isinstance(idx, int):
                if idx < self.num_cells:
                    return self.expression[idx, :], self.covar_vals[idx]
                else:
                    raise IndexError
            else:
                raise NotImplementedError
            

    def __len__(self) -> int:
        return self.num_cells

    def get_dim(self) -> Dict[str, int]:
        return self.num_genes


def data_to_torch_X(X):
    if isinstance(X, sc.AnnData):
        X = X.X
    if not isinstance(X, np.ndarray):
            X = X.toarray()
    return torch.from_numpy(X).float()


def anndata_to_sc_dataset(adata:sc.AnnData, 
                                 species:str="human", 
                                 labels:list=[],
                                 covar_col:str=None,
                                 hv_genes=None,
                                 embedding_model="ESM2",
                                ) -> (SincleCellDataset, AnnData):
    
    # Subset to just genes we have embeddings for
    adata, protein_embeddings = load_gene_embeddings_adata(
        adata=adata,
        species=[species],
        embedding_model=embedding_model
    )
    
    if hv_genes is not None:
        sc.pp.highly_variable_genes(adata, flavor='seurat_v3', n_top_genes=hv_genes)  # Expects Count Data
    
        hv_index = adata.var["highly_variable"]
        adata = adata[:, hv_index] # Subset to hv genes only
    
        protein_embeddings = protein_embeddings[species][hv_index]
    else:
        protein_embeddings = protein_embeddings[species]
    expression = data_to_torch_X(adata.X)
    
    covar_vals = None
    if len(labels) > 0:
        assert covar_col is None or covar_col in labels, "Covar needs to be in labels" # make sure you keep track of covar column!
        labels = adata.obs.loc[:, labels].values
        
        if covar_col is not None:
            # we have a categorical label to use as covariate
            covar_vals = torch.tensor(pd.Categorical(adata.obs[covar_col]).codes)
    return SincleCellDataset(
        expression=expression,
        protein_embeddings=protein_embeddings,
        labels=labels,
        covar_vals=covar_vals
    ), adata    
    
def adata_path_to_prot_chrom_starts(adata, dataset_species, spec_pe_genes, gene_to_chrom_pos, offset):
    """
        Given a :path: to an h5ad, 
    """    
    pe_row_idxs = torch.tensor([spec_pe_genes.index(k.upper()) + offset for k in adata.var_names]).long()
    print(len(np.unique(pe_row_idxs)))
    
    spec_chrom = gene_to_chrom_pos[gene_to_chrom_pos["species"] == dataset_species].set_index("gene_symbol")

    gene_chrom = spec_chrom.loc[[k.upper() for k in adata.var_names]]

    dataset_chroms = gene_chrom["spec_chrom"].cat.codes # now this is correctely indexed by species and chromosome
    print("Max Code:", max(dataset_chroms))
    dataset_pos = gene_chrom["start"].values
    return pe_row_idxs, dataset_chroms, dataset_pos



def process_raw_anndata(row, h5_folder_path, npz_folder_path, scp, skip,
                        additional_filter, root):
        path = row.path
        if not os.path.isfile(root + "/" + path):
            print( "**********************************")
            print(f"***********{root + '/' + path} File Missing****")
            print( "**********************************")
            print(path, root)
            return None

        name = path.replace(".h5ad", "")
        proc_path = path.replace(".h5ad", "_proc.h5ad")
        if skip:
            if os.path.isfile(h5_folder_path + proc_path):
                print(f"{name} already processed. Skipping")
                return None, None, None

        print(f"Proccessing {name}")

        species = row.species
        covar_col = row.covar_col

        ad = sc.read(root + "/" + path)
        labels = []
        if "cell_type" in ad.obs.columns:
            labels.append("cell_type")


        if covar_col is np.nan or np.isnan(covar_col):
            covar_col = None
        else:
            labels.append(covar_col)

        if additional_filter:
            sc.pp.filter_genes(ad, min_cells=10)
            sc.pp.filter_cells(ad, min_genes=25)


        dataset, adata = anndata_to_sc_dataset(ad, species=species, labels=labels, covar_col=covar_col, hv_genes=None)
        adata = adata.copy()

        if additional_filter:
            sc.pp.filter_genes(ad, min_cells=10)
            sc.pp.filter_cells(ad, min_genes=25)
        
        num_cells = adata.X.shape[0]
        num_genes = adata.X.shape[1]

        adata_path = h5_folder_path + proc_path
        adata.write(adata_path)

        arr = data_to_torch_X(adata.X).numpy()

        print(arr.max()) # this is a nice check to make sure it's counts
        filename = npz_folder_path + f"{name}_counts.npz"
        shape = arr.shape
        print(name, shape)
        fp = np.memmap(filename, dtype='int64', mode='w+', shape=shape)
        fp[:] = arr[:]
        fp.flush()
        
        if scp != "":
            subprocess.call(["scp", filename, f"{scp}:{filename}"])
            subprocess.call(["scp", adata_path, f"{scp}:{adata_path}"])
            
        return adata, num_cells, num_genes
    
    
def get_species_to_pe(EMBEDDING_DIR):
    """
    Given an embedding directory, return all embeddings as a dictionary coded by species.
    Note: In the current form, this function is written such that the directory needs all of the following species embeddings.
    """
    EMBEDDING_DIR = Path(EMBEDDING_DIR)

    embeddings_paths = {
            'human': EMBEDDING_DIR / 'Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt',
            'mouse': EMBEDDING_DIR / 'Mus_musculus.GRCm39.gene_symbol_to_embedding_ESM2.pt',
            'frog': EMBEDDING_DIR / 'Xenopus_tropicalis.Xenopus_tropicalis_v9.1.gene_symbol_to_embedding_ESM2.pt',
            'zebrafish': EMBEDDING_DIR / 'Danio_rerio.GRCz11.gene_symbol_to_embedding_ESM2.pt',
            "mouse_lemur": EMBEDDING_DIR / "Microcebus_murinus.Mmur_3.0.gene_symbol_to_embedding_ESM2.pt",
            "pig": EMBEDDING_DIR / 'Sus_scrofa.Sscrofa11.1.gene_symbol_to_embedding_ESM2.pt',
            "macaca_fascicularis": EMBEDDING_DIR / 'Macaca_fascicularis.Macaca_fascicularis_6.0.gene_symbol_to_embedding_ESM2.pt',
            "macaca_mulatta": EMBEDDING_DIR / 'Macaca_mulatta.Mmul_10.gene_symbol_to_embedding_ESM2.pt',
        }
    extra_species = pd.read_csv("./model_files/new_species_protein_embeddings.csv").set_index("species").to_dict()["path"]
    embeddings_paths.update(extra_species) # adds new species
    
    
    
    species_to_pe = {
        species:torch.load(pe_dir) for species, pe_dir in embeddings_paths.items()   
    }

    species_to_pe = {species:{k.upper(): v for k,v in pe.items()} for species, pe in species_to_pe.items()}
    return species_to_pe


def get_spec_chrom_csv(path="/dfs/project/cross-species/yanay/code/all_to_chrom_pos.csv"):
    """
    Get the species to chrom csv file
    """
    gene_to_chrom_pos = pd.read_csv(path)
    gene_to_chrom_pos["spec_chrom"] = pd.Categorical(gene_to_chrom_pos["species"] + "_" +  gene_to_chrom_pos["chromosome"]) # add the spec_chrom list
    return gene_to_chrom_pos