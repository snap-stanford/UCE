import os

# os.environ["NCCL_DEBUG"] = "INFO"
os.environ["OMP_NUM_THREADS"] = "12"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "12"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "12"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "12"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "12"

import warnings

warnings.filterwarnings("ignore")

import scanpy as sc
from tqdm.auto import tqdm
from torch import nn, Tensor

from lit_model import LitUCEModel
from eval_data import MultiDatasetSentences, MultiDatasetSentenceCollator
from utils import figshare_download
import lightning as L
from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.strategies import DeepSpeedStrategy, DDPStrategy
from lightning.pytorch.callbacks import BasePredictionWriter

from torch.utils.data import DataLoader
from data_proc.data_utils import adata_path_to_prot_chrom_starts, \
    get_spec_chrom_csv, process_raw_anndata, get_species_to_pe

import os
import pickle
import pandas as pd
import numpy as np
import torch
import time

class AnndataProcessor:
    def __init__(self, args):
        self.args = args
        
        self.h5_folder_path = self.args.dir
        self.npz_folder_path = self.args.dir
        self.scp = ""

        # Check if paths exist, if not, create them
        self.check_paths()

        # Set up the anndata
        self.adata_name = self.args.adata_path.split("/")[-1]
        self.adata_root_path = self.args.adata_path.replace(self.adata_name, "")
        self.name = self.adata_name.replace(".h5ad", "")
        self.proc_h5_path = self.h5_folder_path + f"{self.name}_proc.h5ad"
        self.adata = None

        # Set up the row
        row = pd.Series()
        row.path = self.adata_name
        row.covar_col = np.nan
        row.species = self.args.species
        self.row = row

        # Set paths once to be used throughout the class
        self.pe_idx_path = self.args.dir + f"{self.name}_pe_idx.torch"
        self.chroms_path = self.args.dir + f"{self.name}_chroms.pkl"
        self.starts_path = self.args.dir + f"{self.name}_starts.pkl"
        self.shapes_dict_path = self.args.dir + f"{self.name}_shapes_dict.pkl"

    def check_paths(self):
        """
        Check if the paths exist, if not, create them
        """
        figshare_download("https://figshare.com/ndownloader/files/42706558",
                                self.args.spec_chrom_csv_path)
        figshare_download("https://figshare.com/ndownloader/files/42706555",
                                self.args.offset_pkl_path)
        if not os.path.exists(self.args.protein_embeddings_dir):
            figshare_download("https://figshare.com/ndownloader/files/42715213",
                'model_files/protein_embeddings.tar.gz')
        figshare_download("https://figshare.com/ndownloader/files/42706585",
                                self.args.token_file)
        if self.args.adata_path is None:
            print("Using sample AnnData: 10k pbmcs dataset")
            self.args.adata_path = "./data/10k_pbmcs_proc.h5ad"
            figshare_download(
                "https://figshare.com/ndownloader/files/42706966",
                self.args.adata_path)
        if self.args.model_loc is None:
            print("Using sample 4 layer model")
            self.args.model_loc = "./model_files/4layer_model.torch"
            figshare_download(
                "https://figshare.com/ndownloader/files/42706576",
                self.args.model_loc)


    def preprocess_anndata(self):
        self.adata, num_cells, num_genes = \
            process_raw_anndata(self.row,
                                self.h5_folder_path,
                                self.npz_folder_path,
                                self.scp,
                                self.args.skip,
                                self.args.filter,
                                root=self.adata_root_path)
        if (num_cells is not None) and (num_genes is not None):
            self.save_shapes_dict(self.name, num_cells, num_genes,
                                   self.shapes_dict_path)

        if self.adata is None:
            self.adata = sc.read(self.proc_h5_path)

    def save_shapes_dict(self, name, num_cells, num_genes, shapes_dict_path):
        shapes_dict = {name: (num_cells, num_genes)}
        with open(shapes_dict_path, "wb+") as f:
            pickle.dump(shapes_dict, f)
            print("Wrote Shapes Dict")

    def generate_idxs(self):
        if os.path.exists(self.pe_idx_path) and \
                os.path.exists(self.chroms_path) and \
                os.path.exists(self.starts_path):
            print("PE Idx, Chrom and Starts files already created")

        else:
            species_to_pe = get_species_to_pe(self.args.protein_embeddings_dir)
            with open(self.args.offset_pkl_path, "rb") as f:
                species_to_offsets = pickle.load(f)

            gene_to_chrom_pos = get_spec_chrom_csv(
                self.args.spec_chrom_csv_path)
            dataset_species = self.args.species
            spec_pe_genes = list(species_to_pe[dataset_species].keys())
            offset = species_to_offsets[dataset_species]
            pe_row_idxs, dataset_chroms, dataset_pos = adata_path_to_prot_chrom_starts(
                self.adata, dataset_species, spec_pe_genes, gene_to_chrom_pos, offset)

            # Save to the temp dict
            torch.save({self.name: pe_row_idxs}, self.pe_idx_path)
            with open(self.chroms_path, "wb+") as f:
                pickle.dump({self.name: dataset_chroms}, f)
            with open(self.starts_path, "wb+") as f:
                pickle.dump({self.name: dataset_pos}, f)

    def run_evaluation(self):
        
        with open(self.shapes_dict_path, "rb") as f:
            shapes_dict = pickle.load(f)
        run_eval(self.adata, self.name, self.pe_idx_path, self.chroms_path,
                 self.starts_path, shapes_dict, self.args)


def get_ESM2_embeddings(args):
    # Load in ESM2 embeddings and special tokens
    all_pe = torch.load(args.token_file)
    if all_pe.shape[0] == 143574:
        torch.manual_seed(23)
        CHROM_TENSORS = torch.normal(mean=0, std=1, size=(1895, args.token_dim))
        # 1895 is the total number of chromosome choices, it is hardcoded for now
        all_pe = torch.vstack(
            (all_pe, CHROM_TENSORS))  # Add the chrom tensors to the end
        all_pe.requires_grad = False

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

def flatten(xss):
    return [x for xs in xss for x in xs]


class CustomWriter(BasePredictionWriter):
    def __init__(self, output_dir, write_interval):
        super().__init__(write_interval)
        self.output_dir = output_dir

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        # this will create N (num processes) files in `output_dir` each containing
        # the predictions of it's respective rank
        torch.save(predictions, os.path.join(self.output_dir, f"embeddings_{trainer.global_rank}.pt"))

        # optionally, you can also save `batch_indices` to get the information about the data index
        # from your prediction data
        torch.save(batch_indices, os.path.join(self.output_dir, f"batch_indices_{trainer.global_rank}.pt"))


def run_eval(adata, name, pe_idx_path, chroms_path, starts_path, shapes_dict,
             args):
    torch.set_float32_matmul_precision('medium')
    #### Set up the model ####
    token_dim = args.token_dim
    emsize = args.emsize  # embedding dimension
    d_hid = args.d_hid  # dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = args.nlayers  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = args.nhead  # number of heads in nn.MultiheadAttention
    dropout = 0.05  # dropout probability
    '''
    model = LitUCEModel(token_dim=args.token_dim,
                         d_model=args.emsize,
                         nhead=args.nhead,
                         d_hid=args.d_hid,
                         nlayers=args.nlayers,
                         output_dim=args.output_dim,
                         dropout=0
                       )
    '''
    if args.model_loc is None:
        raise ValueError("Must provide a model location")
    # intialize as empty
    model = LitUCEModel.load_from_checkpoint(args.model_loc, strict=False)
    # Load in the real token embeddings
    all_pe = get_ESM2_embeddings(args)
    # This will make sure that you don't overwrite the tokens in case you're embedding species from the training data
    # We avoid doing that just in case the random seeds are different across different versions. 
    #if all_pe.shape[0] != 145469: 
    all_pe.requires_grad = False
    #model.pe_embedding = nn.Embedding.from_pretrained(all_pe)
    print(f"Loaded model:\n{args.model_loc}")
    model = model.eval()
    if args.compiled:
        model = torch.compile(model, dynamic=False)
    batch_size = args.batch_size

    #### Run the model ####
    # Dataloaders
    dataset = MultiDatasetSentences(sorted_dataset_names=[name],
                                    shapes_dict=shapes_dict,
                                    args=args, npzs_dir=args.dir,
                                    dataset_to_protein_embeddings_path=pe_idx_path,
                                    datasets_to_chroms_path=chroms_path,
                                    datasets_to_starts_path=starts_path
                                    )
    multi_dataset_sentence_collator = MultiDatasetSentenceCollator(args)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=multi_dataset_sentence_collator,
                            num_workers=1)
    pred_writer = CustomWriter(output_dir=args.dir, write_interval="epoch")
    trainer = L.Trainer(
                        precision="bf16-mixed",
                        strategy=DDPStrategy(process_group_backend="nccl"),
                        num_nodes=args.num_nodes,
                        inference_mode=False, 
                       callbacks=[pred_writer]
                       )

    # or you can set `write_interval="batch"` and override `write_on_batch_end` to save
    # predictions at batch level
    
    trainer.predict(model=model, dataloaders=dataloader, return_predictions=False)

    if trainer.is_global_zero:
        all_written = False # Why is there no wait for all to finish method?
        while not all_written:
            all_written = True
            for rank in range(trainer.world_size):
                for rank in range(trainer.world_size):
                    emb_path = os.path.join(args.dir, f"embeddings_{rank}.pt")
                    idx_path = os.path.join(args.dir, f"batch_indices_{rank}.pt")
    
                    all_written = all_written and os.path.exists(emb_path) & os.path.exists(idx_path)
                if not all_written:
                    time.sleep(0.01)
    if trainer.is_global_zero:
        embeddings = []
        idxs = []
        
        for rank in range(trainer.world_size):
            emb_path = os.path.join(args.dir, f"embeddings_{rank}.pt")
            idx_path = os.path.join(args.dir, f"batch_indices_{rank}.pt")
            rank_emb = torch.load(emb_path, map_location="cpu")
            rank_emb = torch.vstack(rank_emb).numpy()
            embeddings.append(rank_emb)
            rank_idx = flatten(torch.load(idx_path)[0]) # why is this a triple nested list??
            print(rank_emb.shape, len(rank_idx))
            idxs.append(rank_idx)

            os.remove(emb_path)
            os.remove(idx_path)
        dataset_embeds = np.vstack(embeddings)
        idxs = np.concatenate(idxs)
        idxs = np.argsort(idxs)
        adata.obsm["X_uce"] = dataset_embeds[idxs]
        if len(args.append_to_saved_adata) > 0:
            name += ("_" + args.append_to_saved_adata)
        write_path = args.dir + f"{name}_uce_adata.h5ad"
        
        adata.write(write_path)
        
        print("*****Wrote Anndata to:*****")
        print(write_path)
