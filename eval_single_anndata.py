import os
#os.environ["NCCL_DEBUG"] = "INFO"
os.environ["OMP_NUM_THREADS"] = "12" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "12" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "12" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "12" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "12"


import warnings
warnings.filterwarnings("ignore")

import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pickle
import os
import argparse
import logging
import time

from tqdm.auto import tqdm
import pandas as pd
from glob import glob

#sc._settings.ScanpyConfig.n_jobs = 6

import math
from typing import Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset


from accelerate import Accelerator
import anndata

from model import TransformerModel
from eval_data import MultiDatasetSentences, MultiDatasetSentenceCollator

from torch.utils.data import dataset
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import binom

from data_proc.data_utils import adata_path_to_prot_chrom_starts, get_spec_chrom_csv, process_raw_anndata, get_species_to_pe
from pathlib import Path



def get_ESM2_embeddings(args):
    # Load in ESM2 embeddings and special tokens
    all_pe = torch.load(args.token_file)
    if all_pe.shape[0] == 143574:
        torch.manual_seed(23)
        #MASK_TENSOR = torch.normal(mean=0, std=1, size=(1, args.token_dim))
        CHROM_TENSORS = torch.normal(mean=0, std=1, size=(1895, args.token_dim))
        # 1894 is the total number of chromosome choices, it is hardcoded for now
        all_pe = torch.vstack((all_pe, CHROM_TENSORS)) # Add the chrom tensors to the end
        all_pe.requires_grad = False


    #print("Loaded PE", all_pe.shape)
    return all_pe

def padding_tensor(sequences):
    """
    :param sequences: list of tensors
    :return:
    """
    num = len(sequences)
    max_len = max([s.size(0) for s in sequences])
    out_dims = (num, max_len, 1280)
    
    
    out_tensor = sequences[0].data.new(*out_dims).fill_(0)
    out_dims2 = (num, max_len)
    
    mask = sequences[0].data.new(*out_dims2).fill_(float('-inf'))
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        out_tensor[i, :length] = tensor
        mask[i, :length] = 1
    return out_tensor.permute(1, 0, 2), mask


    
MASK_TOKEN_IDX = 0
CHROM_TOKEN_LEFT_IDX = 1
CHROM_TOKEN_RIGHT_IDX = 2
CLS_TOKEN_IDX = 3


def run_eval(adata, name, pe_idx_path, chroms_path, starts_path, shapes_dict, accelerator, args):
    #### Load Files ####
    dataset_to_protein_embeddings = torch.load(pe_idx_path)
    with open(chroms_path, "rb") as f:
        dataset_to_chroms = pickle.load(f)
    with open(starts_path, "rb") as f:
        dataset_to_starts = pickle.load(f)    
    #### Load the model ####
    sample_size = args.sample_size # 256
    num_samples = 1
    output_dim = args.output_dim # 768
     
    token_dim = args.token_dim  # size of vocabulary
    emsize = 1280  # embedding dimension
    d_hid = args.d_hid  # dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = args.nlayers  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 20  # number of heads in nn.MultiheadAttention
    dropout = 0.05  # dropout probability    
    model = TransformerModel(token_dim=token_dim, d_model=emsize, nhead=nhead, d_hid=d_hid, 
                             nlayers=nlayers, dropout=dropout, output_dim=args.output_dim)
    if args.model_loc is None:
        model_loc = f"/dfs/project/cross-species/yanay/code/state_dicts/unwrapped_chrom_model_state_dict_step_{args.step_num}_epoch_{args.epoch_num}_nlayers_{args.nlayers}_sample_size_{args.sample_size}_CLS.torch"
        if args.CXG:
            model_loc += "_CXG"
    else:
        model_loc = args.model_loc
    
    all_pe = get_ESM2_embeddings(args)
    all_pe.requires_grad= False
    model.pe_embedding = nn.Embedding.from_pretrained(all_pe)
    model.load_state_dict(torch.load(model_loc, map_location="cpu"), strict=True)
    print(f"Loaded model:\n{model_loc}")
    model = model.eval()
    model = accelerator.prepare(model)
    token = -1 * torch.ones(token_dim)
    batch_size = args.batch_size
    
    #### Run the model ####
    # Dataloaders
    dataset = MultiDatasetSentences(sorted_dataset_names=[name], shapes_dict=shapes_dict, 
                                    args=args, npzs_dir=args.dir,
                                    dataset_to_protein_embeddings_path=pe_idx_path,
                                    datasets_to_chroms_path=chroms_path,
                                    datasets_to_starts_path=starts_path
                                   )
    multi_dataset_sentence_collator = MultiDatasetSentenceCollator(args)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=multi_dataset_sentence_collator, num_workers=0)
    dataloader = accelerator.prepare(dataloader)
    pbar = tqdm(dataloader, disable=not accelerator.is_local_main_process)
    dataset_embeds = []
    with torch.no_grad():
        for batch in pbar:
            batch_sentences, mask, idxs= batch[0], batch[1], batch[2]
            batch_sentences = batch_sentences.permute(1, 0)
            #print(batch_sentences.squeeze(2).shape)
            batch_sentences = model.module.pe_embedding(batch_sentences.long())
            batch_sentences = nn.functional.normalize(batch_sentences, dim=2) # Normalize token outputs now
            cell_sentences = batch[3]

            _, embedding = model.forward(batch_sentences, mask=mask)
            # Fix for duplicates in last batch
            accelerator.wait_for_everyone()
            embeddings  = accelerator.gather_for_metrics((embedding))
            if accelerator.is_main_process: 
                dataset_embeds.append(embeddings.detach().cpu().numpy())

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        dataset_embeds = np.vstack(dataset_embeds)
        adata.obsm["X_uce"] = dataset_embeds 
        write_path = args.dir + f"{name}_ep_{args.epoch_num}_sn_{args.step_num}_nlayers_{args.nlayers}_sample_size_{args.sample_size}.h5ad"
        adata.write(write_path)
        
        print("*****Wrote Anndata to:*****")
        print(write_path)
    
def main(args, accelerator):
    
    # Step 0 Establish Paths
    adata_name = args.adata_path.split("/")[-1]
    adata_root_path = args.adata_path.replace(adata_name, "")
    h5_folder_path = args.dir # just write these to the same directory
    
    npz_folder_path = args.dir
    scp = ""
    row = pd.Series()
    row.path = adata_name
    row.covar_col = np.nan
    row.species = args.species
    name = adata_name.replace(".h5ad", "")
    proc_h5_path = h5_folder_path + f"{name}_proc.h5ad"
    
    pe_idx_path = args.dir + f"{name}_pe_idx.torch"
    chroms_path = args.dir + f"{name}_chroms.pkl"
    starts_path = args.dir +  f"{name}_starts.pkl"
    shapes_dict_path = args.dir + f"{name}_shapes_dict.pkl"
    adata = None
    
    # STEP 1: Pre Proc the anndata
    if accelerator.is_main_process:
        adata, num_cells, num_genes = process_raw_anndata(row, h5_folder_path, npz_folder_path, 
                                                          scp, args.skip, args.filter, 
                                                          root=adata_root_path)
        # save the shapes dict
        shapes_dict = {name:(num_cells, num_genes)}
        if (num_cells is not None) and (num_genes is not None):
            with open(shapes_dict_path, "wb+") as f:
                pickle.dump(shapes_dict, f)
                print("Wrote Shapes Dict")
            
    # Re read in the anndata if we skipped processing it
    if accelerator.is_main_process:
        if adata is None:
            adata = sc.read(proc_h5_path)
    # STEP 2: Generate the correct idxs
    if accelerator.is_main_process:
        if os.path.exists(pe_idx_path) and os.path.exists(chroms_path) and os.path.exists(starts_path):
            # these have already been created, just read them in
            print("PE Idx, Chrom and Starts files already created")
        else:
            # otherwise, create them first
            species_to_pe = get_species_to_pe(args.protein_embeddings_dir)
            with open(args.offset_pkl_path, "rb") as f:
                species_to_offsets = pickle.load(f)

            gene_to_chrom_pos = get_spec_chrom_csv(args.spec_chrom_csv_path)
            dataset_species = args.species
            spec_pe_genes = list(species_to_pe[dataset_species].keys())
            offset = species_to_offsets[dataset_species]
            pe_row_idxs, dataset_chroms, dataset_pos = adata_path_to_prot_chrom_starts(adata, dataset_species, spec_pe_genes, gene_to_chrom_pos, offset)

            # Save to the temp dict
            torch.save({name:pe_row_idxs}, pe_idx_path)
            with open(chroms_path, "wb+") as f:
                pickle.dump({name:dataset_chroms}, f)
            with open(starts_path, "wb+") as f:
                pickle.dump({name:dataset_pos}, f)    
    
    # STEP 3: Run Evaluation of the model
    accelerator.wait_for_everyone()
    with open(shapes_dict_path, "rb") as f:
        shapes_dict = pickle.load(f)
    run_eval(adata, name, pe_idx_path, chroms_path, starts_path, shapes_dict, accelerator, args)



if __name__=="__main__":
    # Parse command-line arguments
    
    parser = argparse.ArgumentParser(description='Embed a single anndata that has not been processed before using UCE.')
    # Define command-line arguments
    parser.add_argument('--adata_path', type=str, default="/lfs/local/0/yanay/10k_pbmcs.h5ad", help='Full path to the anndata you want to embed.')
    parser.add_argument('--dir', type=str, default="/lfs/local/0/yanay/uce_temp/", help='Working folder where all files will be saved.')
    parser.add_argument('--species', type=str, default="human", help='Species of the anndata.')
    parser.add_argument('--filter', type=bool, default=True, help='Should you do an additional gene/cell filtering on the anndata? This can be a good step since even if you have already done it, subsetting to protein embeddings can make some cells sparser.')
    parser.add_argument('--skip', type=bool, default=True, help='Should you skip datasets that appear to have already been created in the h5 folder?')
        

    # MODEL ARGUMENTS
    parser.add_argument('--batch_size', type=int, help='Set GPU Number')
    parser.add_argument('--CXG', type=bool, help='Use CXG model')    
    parser.add_argument('--epoch_num', type=int, help='Set epoch number')
    parser.add_argument('--step_num', type=int, help='Set Step Number')    
    parser.add_argument('--nlayers', type=int, help='Set Number of layers of transformer')
    parser.add_argument('--output_dim', type=int, help='Set output dim')
    parser.add_argument('--d_hid', type=int, help='Set output dim')
    parser.add_argument('--sample_size', type=int, help='Set sample size of genes')
    parser.add_argument('--token_dim', type=int, help='Set sample size of genes')
    parser.add_argument("--pad_length", type=int, default=None, help="PAD length")
    parser.add_argument("--pad_token_idx", type=int, default=0, help="PAD token index")
    parser.add_argument("--chrom_token_left_idx", type=int, default=1, help="Chrom token left index")
    parser.add_argument("--chrom_token_right_idx", type=int, default=2, help="Chrom token right index")
    parser.add_argument("--cls_token_idx", type=int, default=3, help="CLS token index")
    parser.add_argument("--CHROM_TOKEN_OFFSET", type=int, default=143574, help="Offset index, tokens after this mark are chromosome identifiers")
    
    parser.add_argument('--model_loc', type=str, default=None, help='Specific location of the model if you want to pass a path. Make sure to set all the other model parameters as well as this argument.')    
    
    
    # MISC ARGUMENTS
    parser.add_argument("--spec_chrom_csv_path", default="/dfs/project/cross-species/yanay/code/all_to_chrom_pos.csv", type=str, help="CSV Path for species genes to chromosomes and start locations.")
    parser.add_argument("--token_file", default="/dfs/project/uce/all_species_pe_tokens.torch", type=str, help="Path for token embeddings, torch file.")
    parser.add_argument("--protein_embeddings_dir", default="/dfs/project/cross-species/yanay/data/proteome/embeddings/", type=str, help="Directory where protein embedding .pt files are stored.")
    parser.add_argument("--offset_pkl_path", default="/dfs/project/uce/all_species_offsets.pkl", type=str, help="PKL file which contains offsets for each species.")
    parser.add_argument("--accelerator_dir", default="/lfs/local/0/yanay/mammal_accel_new_dl_chrom_33_eval", type=str, help="Accelerator Dir, not really used.")
    
    
    
    
    parser.set_defaults(
        epoch_num=8,#7,#6,#5,#4,#3,#2,#,1,
        step_num=2013249,#1509937,#1258281,#1006624,#754968,#503312,#251656,
        batch_size=25,
        output_dim=1280,
        d_hid=5120,
        nlayers=33,
        sample_size=1024,
        token_dim=5120,
        pad_length=1536,
        CXG=True
    )
    
    args = parser.parse_args()
    accelerator = Accelerator(project_dir=args.accelerator_dir)
    main(args, accelerator)