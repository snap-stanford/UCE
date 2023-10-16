"""
Utils

"""

import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import sys
import numpy as np


def get_shapes_dict(dataset_path):
    shapes_dict = {}
    datasets_df = pd.read_csv(dataset_path)
    sorted_dataset_names = sorted(datasets_df["names"])

    for name in sorted_dataset_names:
        shapes_dict[name] = (int(datasets_df.set_index("names").loc[name]["num_cells"]), 8000)

    shapes_dict["dev_immune_mouse"] = (443697, 4786)
    shapes_dict["dev_immune_human"] = (34009, 5566)
    shapes_dict["intestinal_tract_human"] =  (69668, 5192)
    shapes_dict["gtex_human"] =  (18511, 7109)
    shapes_dict["gut_endoderm_mouse"] =  (113043, 6806)
    shapes_dict["luca"] =  (249591, 7196)
    shapes_dict.update({
     "madissoon_novel_lung":(190728, 8000),
     'flores_cerebellum_human': (20232, 8000),
     'osuch_gut_human': (272310, 8000),
     'msk_ovarian_human': (929690, 8000),
     'htan_vmuc_dis_epi_human': (65084, 8000),
     'htan_vmuc_val_epi_human': (57564, 8000),
     'htan_vmuc_non_epi_human': (9099, 8000),
     'hao_pbmc_3p_human': (161764, 8000),
     'hao_pbmc_5p_human': (49147, 8000),
     'gao_tumors_human': (36111, 8000),
     'swabrick_breast_human': (92427, 8000),
     'wu_cryo_tumors_human': (105662, 8000),
     'cell_line_het_human': (53513, 8000),
     'bi_allen_metastasis_human': (27787, 8000),
     'zheng68k_human': (68579, 8000),
     'zheng68k_12k_human': (68579, 12000),
     'mouse_embryo_ct': (153597, 12000),
     "regev_gtex_heart": (36574, 8000),
     "tabula_sapiens_heart": (11505, 8000),
     "10k_pbmcs":(11990, 12000),
     "epo_ido":(35834,12000),
     'tabula_sapiens_kidney': (9641, 8000),
     'tabula_microcebus_kidney': (14592, 8000),
     'tabula_muris_kidney': (2781, 8000),
     'tabula_muris_senis_kidney': (19610, 8000),
      'immune_human': (33506, 8000)
                       })

    shapes_dict["zyl_sanes_glaucoma_pig"] = (5901, 6819)
    shapes_dict["parkinsons_macaF"] = (1062, 5103)

    for row in datasets_df.iterrows():
        ngenes = row[1].num_genes
        ncells = row[1].num_cells
        name = row[1].names
        if not np.isnan(ngenes):
            shapes_dict[name] = (int(ncells), int(ngenes))

    return shapes_dict
