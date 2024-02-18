import os

# os.environ["NCCL_DEBUG"] = "INFO"
# os.environ["OMP_NUM_THREADS"] = "12"  # export OMP_NUM_THREADS=4
# os.environ["OPENBLAS_NUM_THREADS"] = "12"  # export OPENBLAS_NUM_THREADS=4
# os.environ["MKL_NUM_THREADS"] = "12"  # export MKL_NUM_THREADS=6
# os.environ["VECLIB_MAXIMUM_THREADS"] = "12"  # export VECLIB_MAXIMUM_THREADS=4
# os.environ["NUMEXPR_NUM_THREADS"] = "12"

import warnings
from dataclasses import dataclass

warnings.filterwarnings("ignore")

import scanpy as sc
from tqdm.auto import tqdm
from torch import nn, Tensor


from torch.utils.data import DataLoader

import os
import pickle
import pandas as pd
import numpy as np
import torch
from typing import Literal, Optional, TypedDict

from uce.model import TransformerModel
from uce.eval_data import MultiDatasetSentences, MultiDatasetSentenceCollator
from uce.utils import figshare_download, gdrive_download
from uce.data_proc.data_utils import adata_path_to_prot_chrom_starts, \
    get_spec_chrom_csv, process_raw_anndata, get_species_to_pe

DOWNLOAD_HOST = "gdrive"

download_urls = {
    "figshare": {
        "spec_chrom_csv": "https://figshare.com/ndownloader/files/42706558",
        "offset_pkl": "https://figshare.com/ndownloader/files/42706555",
        "protein_embeddings": "https://figshare.com/ndownloader/files/42715213",
        "token_file": "https://figshare.com/ndownloader/files/42706585",
        "10k_pbmcs_adata": "https://figshare.com/ndownloader/files/42706966",
        "model4": "https://figshare.com/ndownloader/files/42706576",
    },
    "gdrive": {
        "spec_chrom_csv": "https://drive.google.com/file/d/1vKmaf_Xc14ZlBUTF1duu5ufXtUlfp_fG/view?usp=drive_link",
        "offset_pkl": "https://drive.google.com/file/d/1y20kSv_DfpozOUDhTtaChBRYxXybzGHq/view?usp=drive_link",
        "protein_embeddings": "https://drive.google.com/file/d/1olyRrF5cQX2P2O5or_jKHqZeJZVcgKpT/view?usp=drive_link",
        "token_file": "https://drive.google.com/file/d/1wYDhaO6rIW3X5IJJd7xwyesaOOIYuB57/view?usp=drive_link",
        "10k_pbmcs_adata": "https://drive.google.com/file/d/16ua2ctevSwZB1zhqIf1uffZKk538IMwP/view?usp=drive_link",
        "model4": "https://drive.google.com/file/d/1nQMQcCMtJpyAGmuAf35O4YwWJiYeUC2C/view?usp=drive_link",
    }
}

def download_file(file_tag: str, path: str):
    if DOWNLOAD_HOST == "figshare":
        figshare_download(download_urls["figshare"][file_tag], path)
    elif DOWNLOAD_HOST == "gdrive":
        gdrive_download(download_urls["gdrive"][file_tag], path)
    else:
        raise ValueError(f"Unknown download host: {DOWNLOAD_HOST}")


class AnndataProcessor:
    def __init__(self, args, accelerator):
        self.args = args
        self.accelerator = accelerator
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
        download_file("spec_chrom_csv", self.args.spec_chrom_csv_path)
        download_file("offset_pkl", self.args.offset_pkl_path)
        if not os.path.exists(self.args.protein_embeddings_dir):
            download_file("protein_embeddings", 'model_files/protein_embeddings.tar.gz')
        download_file("token_file", self.args.token_file)
        if self.args.adata_path is None:
            print("Using sample AnnData: 10k pbmcs dataset")
            self.args.adata_path = "./data/10k_pbmcs_proc.h5ad"
            download_file(
                "10k_pbmcs_adata",
                self.args.adata_path)
        if self.args.model_loc is None:
            print("Using sample 4 layer model")
            self.args.model_loc = "./model_files/4layer_model.torch"
            download_file(
                "model4",
                self.args.model_loc)


    def preprocess_anndata(self):
        if self.accelerator is None or self.accelerator.is_main_process:
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
        if self.accelerator is None or self.accelerator.is_main_process:
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
        self.accelerator.wait_for_everyone()
        with open(self.shapes_dict_path, "rb") as f:
            shapes_dict = pickle.load(f)
        run_eval(self.adata, self.name, self.pe_idx_path, self.chroms_path,
                 self.starts_path, shapes_dict, self.accelerator, self.args)


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

def get_eval_model(args):
    emsize = 1280  # embedding dimension
    nhead = 20  # number of heads in nn.MultiheadAttention
    dropout = 0.05  # dropout probability
    token_dim = args.token_dim
    d_hid = args.d_hid  # dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = args.nlayers  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    model = TransformerModel(
        token_dim=token_dim,
        d_model=emsize,
        nhead=nhead,
        d_hid=d_hid,
        nlayers=nlayers,
        dropout=dropout,
        output_dim=args.output_dim
    )
    if args.model_loc is None:
        raise ValueError("Must provide a model location")

    # intialize as empty
    empty_pe = torch.zeros(145469, 5120, requires_grad=False)
    # empty_pe.requires_grad = False
    model.pe_embedding = nn.Embedding.from_pretrained(empty_pe)
    model.load_state_dict(torch.load(args.model_loc, map_location="cpu"),
                          strict=True)
    # Load in the real token embeddings
    all_pe = get_ESM2_embeddings(args)
    # This will make sure that you don't overwrite the tokens in case you're embedding species from the training data
    # We avoid doing that just in case the random seeds are different across different versions. 
    if all_pe.shape[0] != 145469: 
        all_pe.requires_grad = False
        model.pe_embedding = nn.Embedding.from_pretrained(all_pe)
    model.eval()
    return model
    


def run_eval(adata, name, pe_idx_path, chroms_path, starts_path, shapes_dict,
             accelerator, args):

    #### Set up the model ####
    token_dim = args.token_dim
    emsize = 1280  # embedding dimension
    d_hid = args.d_hid  # dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = args.nlayers  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 20  # number of heads in nn.MultiheadAttention
    dropout = 0.05  # dropout probability
    model = TransformerModel(token_dim=token_dim, d_model=emsize, nhead=nhead,
                             d_hid=d_hid,
                             nlayers=nlayers, dropout=dropout,
                             output_dim=args.output_dim)
    if args.model_loc is None:
        raise ValueError("Must provide a model location")
    # intialize as empty
    empty_pe = torch.zeros(145469, 5120)
    empty_pe.requires_grad = False
    model.pe_embedding = nn.Embedding.from_pretrained(empty_pe)
    model.load_state_dict(torch.load(args.model_loc, map_location="cpu"),
                          strict=True)
    # Load in the real token embeddings
    all_pe = get_ESM2_embeddings(args)
    # This will make sure that you don't overwrite the tokens in case you're embedding species from the training data
    # We avoid doing that just in case the random seeds are different across different versions. 
    if all_pe.shape[0] != 145469: 
        all_pe.requires_grad = False
        model.pe_embedding = nn.Embedding.from_pretrained(all_pe)
    print(f"Loaded model:\n{args.model_loc}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.eval().to(device)
    model = accelerator.prepare(model)
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
                            pin_memory=True,
                            num_workers=0)
    dataloader = accelerator.prepare(dataloader)
    pbar = tqdm(dataloader, disable=not accelerator.is_local_main_process)
    dataset_embeds = []
    with torch.no_grad():
        for batch in pbar:
            batch_sentences, mask, idxs = batch[0], batch[1], batch[2]
            batch_sentences = batch_sentences.permute(1, 0).to(device)
            if args.multi_gpu:
                batch_sentences = model.module.pe_embedding(batch_sentences.long())
            else:
                batch_sentences = model.pe_embedding(batch_sentences.long())
            batch_sentences = nn.functional.normalize(batch_sentences,
                                                      dim=2)  # Normalize token outputs now
            _, embedding = model.forward(batch_sentences, mask=mask.to(device))
            # Fix for duplicates in last batch
            accelerator.wait_for_everyone()
            embeddings = accelerator.gather_for_metrics((embedding))
            if accelerator.is_main_process:
                dataset_embeds.append(embeddings.detach().cpu().numpy())

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        dataset_embeds = np.vstack(dataset_embeds)
        adata.obsm["X_uce"] = dataset_embeds
        write_path = args.dir + f"{name}_uce_adata.h5ad"
        adata.write(write_path)

        print("*****Wrote Anndata to:*****")
        print(write_path)

def get_pretrained_model(size: Literal['small', 'large']) -> TransformerModel:
    assert size in ('small', 'large'), "size parameter must be either 'small' or 'large'"
    if size == 'large':
        raise NotImplementedError('Automatic fetching of the large model is not implemented yet')
    print('Loading small pretrained model with 4 layers...')
    from uce.eval_single_anndata import parser
    default_args = parser.parse_args([])
    if default_args.model_loc is None:
        print("Using sample 4 layer model")
        default_args.model_loc = "./model_files/4layer_model.torch"
        download_file(
            "model4",
            default_args.model_loc)
    return get_eval_model(default_args)


@dataclass
class DatasetFile:
    """Corresponds to flags in the `uce-eval-single-anndata` cli"""
    adata_path: Optional[str] = None
    species: str = 'human'
    filter: bool = True
    skip: bool = True

class DatasetFileDict(TypedDict, total=False):
    adata_path: str
    species: str
    filter: bool
    skip: bool


def get_processed_dataset(dataset_file: Optional[DatasetFile | DatasetFileDict] = None, batch_size: int = 1) -> tuple[MultiDatasetSentences, DataLoader]:
    from uce.eval_single_anndata import parser
    args = parser.parse_args([])
    if dataset_file is not None:
        if isinstance(dataset_file, dict):
            dataset_file = DatasetFile(**dataset_file)
        args.adata_path = dataset_file.adata_path
        args.species = dataset_file.species
        args.filter = dataset_file.filter
        args.skip = dataset_file.skip

    processor = AnndataProcessor(args=args, accelerator=None)
    processor.preprocess_anndata()
    processor.generate_idxs()
    with open(processor.shapes_dict_path, "rb") as f:
        shapes_dict = pickle.load(f)

    dataset = MultiDatasetSentences(sorted_dataset_names=[processor.name],
                                    shapes_dict=shapes_dict,
                                    args=args, npzs_dir=args.dir,
                                    dataset_to_protein_embeddings_path=processor.pe_idx_path,
                                    datasets_to_chroms_path=processor.chroms_path,
                                    datasets_to_starts_path=processor.starts_path
                                    )
    multi_dataset_sentence_collator = MultiDatasetSentenceCollator(args)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=multi_dataset_sentence_collator,
                            num_workers=10, pin_memory=True)
    return dataset, dataloader