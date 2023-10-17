"""
Script for evaluating a single anndata

"""

import argparse
from evaluate import AnndataProcessor
from accelerate import Accelerator

def main(args, accelerator):
    processor = AnndataProcessor(args, accelerator)
    processor.preprocess_anndata()
    processor.generate_idxs()
    processor.run_evaluation()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Embed a single anndata using UCE.')

    # Anndata Processing Arguments
    parser.add_argument('--adata_path', type=str,
                        default="./data/10k_pbmcs.h5ad",
                        help='Full path to the anndata you want to embed.')
    parser.add_argument('--dir', type=str,
                        default="./",
                        help='Working folder where all files will be saved.')
    parser.add_argument('--species', type=str, default="human",
                        help='Species of the anndata.')
    parser.add_argument('--filter', type=bool, default=True,
                        help='Additional gene/cell filtering on the anndata.')
    parser.add_argument('--skip', type=bool, default=True,
                        help='Skip datasets that appear to have already been created.')

    # Model Arguments
    parser.add_argument('--model_loc', type=str,
                        default="model_files/4layer_model.torch",
                        help='Location of the model.')
    parser.add_argument('--batch_size', type=int, default=25,
                        help='Batch size.')
    parser.add_argument('--CXG', type=bool, default=True,
                        help='Use CXG model.')
    parser.add_argument('--nlayers', type=int, default=4,
                        help='Number of transformer layers.')
    parser.add_argument('--output_dim', type=int, default=1280,
                        help='Output dimension.')
    parser.add_argument('--d_hid', type=int, default=5120,
                        help='Hidden dimension.')
    parser.add_argument('--token_dim', type=int, default=5120,
                        help='Token dimension.')

    # Misc Arguments
    parser.add_argument("--spec_chrom_csv_path",
                        default="./model_files/species_chrom.csv", type=str,
                        help="CSV Path for species genes to chromosomes and start locations.")
    parser.add_argument("--token_file",
                        default="./model_files/all_tokens.torch", type=str,
                        help="Path for token embeddings.")
    parser.add_argument("--protein_embeddings_dir",
                        default="./model_files/protein_embeddings/", type=str,
                        help="Directory where protein embedding .pt files are stored.")
    parser.add_argument("--offset_pkl_path",
                        default="./model_files/species_offsets.pkl", type=str,
                        help="PKL file which contains offsets for each species.")

    args = parser.parse_args()
    accelerator = Accelerator(project_dir=args.accelerator_dir)
    main(args, accelerator)