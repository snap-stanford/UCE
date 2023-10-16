"""
Dataloaders

"""

import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append('../')
from typing import Dict, List, Optional, Tuple, Any
import torch
import numpy as np
import pickle
import torch.utils.data as data


class MultiDatasetSentences(data.Dataset):
    def __init__(self, sorted_dataset_names, shapes_dict, args, 
                 dataset_to_protein_embeddings_path= "/lfs/local/0/yanay/reduced_datasets_to_pe_chrom_5120_new.torch",
                 datasets_to_chroms_path="/lfs/local/0/yanay/dataset_to_chroms_new.pkl",
                 datasets_to_starts_path="/lfs/local/0/yanay/dataset_to_starts_new.pkl",
                 npzs_dir="/lfs/local/0/yanay/uce_proc/") -> None:
        super(MultiDatasetSentences, self).__init__()
        # self.xs = {}
        self.num_cells = {}
        self.num_genes = {}
        self.shapes_dict = shapes_dict
        self.args = args

        self.total_num_cells = 0
        for name in sorted_dataset_names:
            num_cells, num_genes = self.shapes_dict[name]
            # self.xs[name] = X
            self.num_cells[name] = num_cells
            self.num_genes[name] = num_genes

            self.total_num_cells += num_cells

        self.datasets = sorted_dataset_names

        # TODO: preferably not hard-coded here
        self.dataset_to_protein_embeddings = torch.load(dataset_to_protein_embeddings_path)
        with open(datasets_to_chroms_path, "rb") as f:
            self.dataset_to_chroms = pickle.load(f)
        with open(datasets_to_starts_path, "rb") as f:
            self.dataset_to_starts = pickle.load(f)
        
        self.npzs_dir = npzs_dir

    def __getitem__(self, idx):
        if isinstance(idx, int):
            for dataset in sorted(self.datasets):
                if idx < self.num_cells[dataset]:
                    #cts = np.memmap(f"/lfs/local/0/yanay/cxg_npzs/" + f"{dataset}_counts.npz",
                    #        dtype='int64', mode='r', shape=self.shapes_dict[dataset])
                    cts = np.memmap(self.npzs_dir + f"{dataset}_counts.npz", dtype='int64', mode='r', shape=self.shapes_dict[dataset])
                    counts = cts[idx]
                    counts = torch.tensor(counts).unsqueeze(0)
                    weights = torch.log1p(counts)
                    weights = (weights / torch.sum(weights))
                    batch_sentences, mask, seq_len, cell_sentences = \
                        sample_cell_sentences(counts, weights, dataset, self.args,
                            dataset_to_protein_embeddings= self.dataset_to_protein_embeddings,
                            dataset_to_chroms=self.dataset_to_chroms,
                            dataset_to_starts=self.dataset_to_starts)
                    return batch_sentences, mask, idx, seq_len, cell_sentences
                else:
                    idx -= self.num_cells[dataset]
            raise IndexError
        else:
            raise NotImplementedError

    def __len__(self) -> int:
        return self.total_num_cells

    def get_dim(self) -> Dict[str, int]:
        return self.num_genes


class MultiDatasetSentenceCollator(object):
    def __init__(self, args):
        self.pad_length = args.pad_length


    def __call__(self, batch):
        batch_size = len(batch)
        batch_sentences = torch.zeros((batch_size, self.pad_length))
        mask = torch.zeros((batch_size, self.pad_length))
        cell_sentences = torch.zeros((batch_size, self.pad_length))

        idxs = torch.zeros(batch_size)

        i = 0
        max_len = 0
        for bs, msk, idx, seq_len, cs in batch:
            batch_sentences[i, :] = bs
            cell_sentences[i, :] = cs
            max_len = max(max_len, seq_len)
            mask[i, :] = msk
            idxs[i] = idx

            i += 1

        return batch_sentences[:, :max_len] , mask[:, :max_len], idxs, cell_sentences



def sample_cell_sentences(counts, batch_weights, dataset, args,
                          dataset_to_protein_embeddings,
                          dataset_to_chroms,
                          dataset_to_starts):

    dataset_idxs = dataset_to_protein_embeddings[dataset]
    cell_sentences = torch.zeros((counts.shape[0], args.pad_length))
    # pos = adata.X > 0
    mask = torch.zeros((counts.shape[0], args.pad_length))

    chroms = dataset_to_chroms[dataset]
    starts = dataset_to_starts[dataset]

    longest_seq_len = 0

    for c, cell in enumerate(counts):
        pos_genes = torch.where(counts[c] > 0)[0]
        neg_genes = torch.where(counts[c] < 1)[0]
        if len(pos_genes) == 0:
            pos_genes = neg_genes

        weights = batch_weights[c].numpy()
        weights = weights / sum(weights)  # RE NORM after mask

        # mask.append(torch.ones(sample_size))
        choice_idx = np.random.choice(np.arange(len(weights)),
                                      size=args.sample_size, p=weights,
                                      replace=True)
        choosen_chrom = chroms[choice_idx]
        chrom_sort = np.argsort(choosen_chrom)  # order by chromsome
        choice_idx = choice_idx[chrom_sort]  # now ordered by chrom

        # sort by start
        new_chrom = chroms[choice_idx]
        choosen_starts = starts[choice_idx]

        ordered_choice_idx = np.full((args.pad_length),
                                     args.cls_token_idx)  # start with cls
        # i= 0 first token is CLS
        i = 1  # continue on to the rest of the sequence with left bracket being assumed.\
        # Shuffle the chroms now
        uq_chroms = np.unique(new_chrom)
        np.random.shuffle(uq_chroms) # shuffle
        for chrom in uq_chroms:
            # Open Chrom
            ordered_choice_idx[i] = int(chrom) + args.CHROM_TOKEN_OFFSET # token of this chromosome # i = 1 next token is a chrom open
            i += 1
            # now sort the by start order within the chroms
            loc = np.where(new_chrom == chrom)[0]
            sort_by_start = np.argsort(
                choosen_starts[loc])  # start locations for these chromsomes

            to_add = choice_idx[loc[sort_by_start]]
            ordered_choice_idx[i:(i + len(to_add))] = dataset_idxs[to_add]  # convert
            i += len(to_add)
            ordered_choice_idx[i] = args.chrom_token_right_idx # add the chrom sep again
            i += 1  # add the closing token again

        longest_seq_len = max(longest_seq_len, i)
        remainder_len = (args.pad_length - i)

        cell_mask = torch.concat((torch.ones(i),
                                  # pay attention to all of these tokens, ignore the rest!
                                  torch.zeros(remainder_len)))

        mask[c, :] = cell_mask

        ordered_choice_idx[i:] = args.pad_token_idx  # mask

        # sample_row = pos_gene_embeds[choice_idx, :]
        cell_sentences[c, :] = torch.from_numpy(ordered_choice_idx)
        
    cell_sentences_pe = cell_sentences.long()  # .unsqueeze(2) # all_pe[cell_sentences.long(), :]
    
    return cell_sentences_pe, mask, longest_seq_len, cell_sentences