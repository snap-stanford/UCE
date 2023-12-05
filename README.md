# Universal Cell Embeddings

This repo includes a PyTorch [HuggingFace Accelerator](https://huggingface.co/docs/accelerate/package_reference/accelerator) implementation of the UCE model, to be used to embed individual anndata datasets.

## Installation

```
pip install -r requirements.txt
```

## Embedding a new dataset

To generate an embedding for a new single-cell RNA sequencing dataset in the AnnData format, use the `eval_single_anndata.py` script.

```
python eval_single_anndata.py --adata_path {path_to_anndata} --dir {output_dir} --species {species} --model_loc {model_loc} --batch_size {batch_size}
```

where
- `adata_path`: a h5ad file. The `.X` slot of the file should be scRNA-seq counts. The `.var_names` slot should correspond to gene names, *not ENSEMBLIDs*.
- `dir`: the working directory in which intermediate and final output files will be saved to skip repeated processing of the same dataset.
- `species`: the species of the dataset you are embedding.
- `model_loc`: the location of the model weights `.torch` file.
- `batch_size`: the per GPU batch size. For the 33 layer model, on a 80GB GPU, you should use 25. For a 4 layer model on the same GPU, you can use 100.

For a sample output on the 10k pbmc dataset, run
```
python eval_single_anndata.py
```
All necessary model files will be downloaded automatically.


**Note**: This script makes use of additional files, which are described in the code documentation. These are downloaded automatically unless already present in the working directory. The script defaults to the pretrained 4-layer model. For running the pretrained 33-layer model from the paper, please download using this [link](https://figshare.com/articles/dataset/Universal_Cell_Embedding_Model_Files/24320806?file=43423236) and set `--nlayers 33`.

## Output

Final evaluated AnnData: `dir/{dataset_name}.h5ad`. This AnnData will be 
identical to the proccessed input anndata, but have UCE embeddings added in the `.obsm["X_uce"]` slot.

Please see documentation for information on additional output files. All 
outputs from `eval_single_anndata.py` are stored in the `dir` directory.

## Data

You can download processed datasets used in the papere [here](https://drive.google.com/drive/folders/1f63fh0ykgEhCrkd_EVvIootBw7LYDVI7?usp=drive_link)

**Note:** These datasets were embedded using the 33 layer model. Embeddings for the 33 layer model are not compatible with embeddings from the 4 layer model.

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
