# Universal Cell Embeddings

This repo includes a PyTorch [HuggingFace Accelerator](https://huggingface.co/docs/accelerate/package_reference/accelerator) implementation of the UCE model, to be used to embed individual anndata datasets.

## Installation
UCE can be installed from PyPI using pip:

```sh
python -m pip install uce-model
```

## Embedding a new dataset

To generate an embedding for a new single-cell RNA sequencing dataset in the AnnData format, use the `uce-eval-single-anndata` command (depending on your environment you might need to run `python -m uce-eval-single-anndata`, so we will write this below).

```
python -m uce-eval-single-anndata --adata_path {path_to_anndata} --dir {output_dir} --species {species} --model_loc {model_loc} --batch_size {batch_size}
```

where
- `adata_path`: a h5ad file. The `.X` slot of the file should be scRNA-seq counts. The `.var_names` slot should correspond to gene names, *not ENSEMBLIDs*.
- `dir`: the working directory in which intermediate and final output files will be saved to skip repeated processing of the same dataset.
- `species`: the species of the dataset you are embedding.
- `model_loc`: the location of the model weights `.torch` file.
- `batch_size`: the per GPU batch size. For the 33 layer model, on a 80GB GPU, you should use 25. For a 4 layer model on the same GPU, you can use 100.

For a sample output on the 10k pbmc dataset, run
```
python -m uce-eval-single-anndata
```
All necessary model files will be downloaded automatically.


Details on all the options for `uce-eval-single-anndata` can be found by running `python -m uce-eval-single-anndata --help`.

<details>
<summary> Output from <code>uce-eval-single-anndata --help</code> </summary>

```
usage: uce-eval-single-anndata [-h] [--adata_path ADATA_PATH] [--dir DIR] [--species SPECIES] [--filter FILTER]
                               [--skip SKIP] [--model_loc MODEL_LOC] [--batch_size BATCH_SIZE]
                               [--pad_length PAD_LENGTH] [--pad_token_idx PAD_TOKEN_IDX]
                               [--chrom_token_left_idx CHROM_TOKEN_LEFT_IDX]
                               [--chrom_token_right_idx CHROM_TOKEN_RIGHT_IDX] [--cls_token_idx CLS_TOKEN_IDX]
                               [--CHROM_TOKEN_OFFSET CHROM_TOKEN_OFFSET] [--sample_size SAMPLE_SIZE] [--CXG CXG]
                               [--nlayers NLAYERS] [--output_dim OUTPUT_DIM] [--d_hid D_HID] [--token_dim TOKEN_DIM]
                               [--multi_gpu MULTI_GPU] [--spec_chrom_csv_path SPEC_CHROM_CSV_PATH]
                               [--token_file TOKEN_FILE] [--protein_embeddings_dir PROTEIN_EMBEDDINGS_DIR]
                               [--offset_pkl_path OFFSET_PKL_PATH]

Embed a single anndata using UCE.

options:
  -h, --help            show this help message and exit
  --adata_path ADATA_PATH
                        Full path to the anndata you want to embed. (default: None)
  --dir DIR             Working folder where all files will be saved. (default: ./)
  --species SPECIES     Species of the anndata. (default: human)
  --filter FILTER       Additional gene/cell filtering on the anndata. (default: True)
  --skip SKIP           Skip datasets that appear to have already been created. (default: True)
  --model_loc MODEL_LOC
                        Location of the model. (default: None)
  --batch_size BATCH_SIZE
                        Batch size. (default: 25)
  --pad_length PAD_LENGTH
                        Batch size. (default: 1536)
  --pad_token_idx PAD_TOKEN_IDX
                        PAD token index (default: 0)
  --chrom_token_left_idx CHROM_TOKEN_LEFT_IDX
                        Chrom token left index (default: 1)
  --chrom_token_right_idx CHROM_TOKEN_RIGHT_IDX
                        Chrom token right index (default: 2)
  --cls_token_idx CLS_TOKEN_IDX
                        CLS token index (default: 3)
  --CHROM_TOKEN_OFFSET CHROM_TOKEN_OFFSET
                        Offset index, tokens after this mark are chromosome identifiers (default: 143574)
  --sample_size SAMPLE_SIZE
                        Number of genes sampled for cell sentence (default: 1024)
  --CXG CXG             Use CXG model. (default: True)
  --nlayers NLAYERS     Number of transformer layers. (default: 4)
  --output_dim OUTPUT_DIM
                        Output dimension. (default: 1280)
  --d_hid D_HID         Hidden dimension. (default: 5120)
  --token_dim TOKEN_DIM
                        Token dimension. (default: 5120)
  --multi_gpu MULTI_GPU
                        Use multiple GPUs (default: False)
  --spec_chrom_csv_path SPEC_CHROM_CSV_PATH
                        CSV Path for species genes to chromosomes and start locations. (default:
                        ./model_files/species_chrom.csv)
  --token_file TOKEN_FILE
                        Path for token embeddings. (default: ./model_files/all_tokens.torch)
  --protein_embeddings_dir PROTEIN_EMBEDDINGS_DIR
                        Directory where protein embedding .pt files are stored. (default:
                        ./model_files/protein_embeddings/)
  --offset_pkl_path OFFSET_PKL_PATH
                        PKL file which contains offsets for each species. (default:
                        ./model_files/species_offsets.pkl)
```

</details>
<br/>


**Note**: This script makes use of additional files, which are described in the code documentation. These are downloaded automatically unless already present in the working directory. The script defaults to the pretrained 4-layer model. For running the pretrained 33-layer model from the paper, please download using this [link](https://figshare.com/articles/dataset/Universal_Cell_Embedding_Model_Files/24320806?file=43423236) and set `--nlayers 33`.

## Output

Final evaluated AnnData: `dir/{dataset_name}.h5ad`. This AnnData will be 
identical to the proccessed input anndata, but have UCE embeddings added in the `.obsm["X_uce"]` slot.

Please see documentation for information on additional output files. All 
outputs from `uce-eval-single-anndata` are stored in the `dir` directory.

## Data

You can download processed datasets used in the papere [here](https://drive.google.com/drive/folders/1f63fh0ykgEhCrkd_EVvIootBw7LYDVI7?usp=drive_link)

**Note:** These datasets were embedded using the 33 layer model. Embeddings for the 33 layer model are not compatible with embeddings from the 4 layer model.


## Using the model in Python
UCE is also a Python library, and the model can be loaded as a PyTorch module.

```python
import uce

model = uce.get_pretrained('small')  # 'small' gets a 4-layer model, 'large' gets a 33-layer model
```

You can also set up a dataloader and a dataset to embed a new dataset using the model.
```python

```

## Citing

If you find our paper and code useful, please consider citing the [preprint](https://www.biorxiv.org/content/10.1101/2023.11.28.568918v1):

```
@article{rosen2023universal,
  title={Universal Cell Embeddings: A Foundation Model for Cell Biology},
  author={Rosen, Yanay and Roohani, Yusuf and Agrawal, Ayush and Samotorcan, Leon and Consortium, Tabula Sapiens and Quake, Stephen R and Leskovec, Jure},
  journal={bioRxiv},
  pages={2023--11},
  year={2023},
  publisher={Cold Spring Harbor Laboratory}
}
```
