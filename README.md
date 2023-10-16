# Universal Cell Embeddings

This repo includes a PyTorch [HuggingFace Accelerator](https://huggingface.co/docs/accelerate/package_reference/accelerator) implementation of the UCE model, to be used to embed individual anndata datasets.

To evaluate an anndata dataset, use the `eval_single_anndata.py` script.

To run the script, you need the following files:

- An AnnData h5ad file:
  
- The pretrained UCE model's weights, in a `.torch` file
- The token embeddings for the UCE model, in a `.torch` file
- A `.pkl` file containing information that links each species to the corresponding tokens in the token embedding file.
- A csv containing information about the chromosomes for each species


## Setting up UCE

### Protein Embeddings
To run UCE you will need protein embeddings. We will provide a download link to a protein embeddings directory.


### Requirements

UCE requires installation of a number of python modules. Please install them via:

```
pip install -r requirements.txt
```

## Running UCE

To run UCE, use the `eval_single_anndata.py` file.

`eval_single_anndata.py` accepts a number of required arguments:

- `adata_path`: a h5ad file.
The `.X` slot of the file should be scRNA-seq counts. The `.var_names` slot should correspond to gene names, *not ENSEMBLIDs*.

- `dir`: the working directory in which intermediate and final output files will be saved. This directory is checked for existing temp files, which can be used to skip repeated processing of the same dataset.
- `species`: the species of the dataset you are embedding.
- `batch_size`: the per GPU batch size. For the 33 layer model, on a 80GB GPU, you should use 25. For a 4 layer model on the same GPU, you can use 100.
- `nlayers`: the number of layers of the model. This should be specified even if you are loading a specific model file already.
- `model_loc`: the location of the model weights `.torch` file.


There are also a few file paths you need to specify. You might want to edit the source code to make these the default locations instead of specifiying these everytime.

- `spec_chrom_csv_path`:  This is a csv file mapping genes from each species to their respective chromosomes and genomic start positions.
- `token_file`: This is a `.torch` file containing protein embeddings for all tokens.
- `protein_embeddings_dir`: This directory contains protein embedding `.pt` files for all species. 
- `offset_pkl_path`: This `.pkl` file maps between species and their gene's locations in the `token_file`.

- `accelerator_dir`: This directory should just be an existing directory somewhere on the machine. It's required for accelerator to work, but since we're not doing any model training it's not useful.


## UCE Output

`eval_single_anndata.py` will output a number of files.

All of these files will be in the same directory, the `dir` directory.

The name of the dataset, `{dataset_name}` you are evaluating will be taken as the file name from `adata_path`, minus the `.h5ad` suffix.

**AnnDatas:**
- Final evaluated AnnData: `dir/{dataset_name}.h5ad`

This AnnData will be identical to the proccessed input anndata, but have UCE embeddings added in the `.obsm["X_uce"]` slot.

**Proccessed Files:**
- `dir/{dataset_name}_proc.h5ad`: the proccessed anndata. Proccessing the anndata means subsetting it to genes which have protein embeddings and then refiltering the dataset by minimum counts. 
- `dir/{dataset_name}_chroms.pkl`: This file maps the genes in the dataset to their corresponding chromosome idxs.
- `dir/{dataset_name}_counts.npz`: This file contains the counts of the anndata in a easy to access format.
- `dir/{dataset_name}_shapes_dict.pkl`: This file contains the shape (ncell x ngene) of the anndata, used to read the `.npz` file.
- `dir/{dataset_name}_pe_idx.torch`: This file maps between the genes in the dataset and their index in the tokens file.
- `dir/{dataset_name}_starts.pkl`: This file maps between the genes in the dataset and their genomic start locations.



The pretraining AnnData has the same format the final AnnData.

**Final Macrogene Weights:**
- Gene to macrogene final weights file: `{run_name}_genes_to_macrogenes.pkl`

**Log Files:**

There are a number of additional log files outputted:
- `{run_name}_triplets.csv` A csv with information about which triplets were mined during metric learning
- `{run_name}_epoch_scores.csv` A csv with information about scoring during metric learning
- `{run_name}_celltype_id.pkl` A pkl of a dictionary containing cell type to categorical codings used for interpreting the other log files
