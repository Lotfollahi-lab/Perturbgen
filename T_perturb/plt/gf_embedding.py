import os
import re

import numpy as np
import scanpy as sc
import torch
from matplotlib import pyplot as plt
from matplotlib import style
from src.utils import non_sorted_EmbExtractor, read_dataset_files

style.use('default')
style.use(
    '/lustre/scratch123/hgi/projects/healthy_imm_expr/'
    't_generative/T_perturb/T_perturb/pp/mpl_style.mplstyle'
)


# Set default figure facecolor to white
plt.rcParams['figure.facecolor'] = 'white'
if os.getcwd().split('/')[-3] != 'T_perturb':
    # set working directory to root of repository
    os.chdir(
        '/lustre/scratch123/hgi/projects/healthy_imm_expr/'
        't_generative/T_perturb/T_perturb/plt'
    )
    print('Changed working directory to root of repository')

tokenized_dir = (
    '/lustre/scratch123/hgi/projects/healthy_imm_expr/'
    't_generative/T_perturb/T_perturb/pp/res/dataset_hvg_tgt'
)
adata_dir = (
    '/lustre/scratch123/hgi/projects/healthy_imm_expr/'
    't_generative/T_perturb/T_perturb/pp/res/h5ad_pairing_hvg_tgt'
)


dataset_dict = read_dataset_files(tokenized_dir, 'dataset')
dataset_dict_sorted = {k: dataset_dict[k] for k in sorted(dataset_dict)}
adata_dict = read_dataset_files(adata_dir, 'h5ad')

dataset_keys = list(dataset_dict_sorted.keys())


def extract_number_or_inf(s):
    match = re.search(r'\d+', s)
    if match is not None:
        return int(match.group())
    return float('inf')


# Sort dataset_dict_keys
dataset_dict_keys = sorted(dataset_keys, key=extract_number_or_inf)

# Assuming adata_dict.keys() is similar to dataset_keys
adata_keys = list(adata_dict.keys())
adata_dict_keys = sorted(adata_keys, key=extract_number_or_inf)
# sort dictionnary based on naming

dataset_path = os.listdir(tokenized_dir)
for i in range(len(dataset_dict_keys)):
    adata = adata_dict[adata_dict_keys[i]]
    print(adata_dict_keys[i])
    dataset = dataset_dict_sorted[dataset_dict_keys[i]]
    print(dataset_dict_keys[i])
    print(dataset_path[i])
    num_labels = len(set(dataset['Cell_type']))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    embex = non_sorted_EmbExtractor(
        model_type='CellClassifier',
        num_classes=num_labels,  # number of cell types for unsupervised training
        emb_mode='cell',
        max_ncells=len(dataset),  # extract embeddings for all cells
        emb_layer=0,  # 0 = to last layer, -1 = second to last layer
        forward_batch_size=64,
        emb_label=['Cell_culture_batch', 'Cell_type'],
        labels_to_plot=['Cell_culture_batch', 'Cell_type'],
        nproc=64,
        summary_stat=None,
    )

    # embs = embex.extract_embs(
    #     './res/Geneformer/240131_geneformer_CellClassifier_L2048_B32_LR5e-05_LSlinear_WU10000_E3_Oadamw_F5_16h/checkpoint-16272',
    #     '/lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/'
    #     'T_perturb/T_perturb/pp/res/dataset/cytoimmgen_degs_random_pairing_16h.dataset',
    #     './res/Geneformer',
    #     'cell_embeddings_finetuned_16h',
    # )
    time_point = np.unique(dataset['Time_point']).item()
    embs = embex.extract_embs(
        '/lustre/scratch123/hgi/projects/healthy_imm_expr/'
        't_generative/generative_modelling_omic/Geneformer',
        f'{tokenized_dir}/{dataset_path[i]}',
        './res/Geneformer',
        f'cell_embeddings_zeroshot_{time_point}',
    )

    # only keep numerical columns from embs
    embs_embeddings = embs.select_dtypes(include='number')
    assert adata.obs['Cell_type'].tolist() == embs['Cell_type'].tolist()

    # convert embs into np.array
    embs_embeddings = embs_embeddings.to_numpy()
    adata.obsm['X_GF_zero_shot'] = embs_embeddings
    np.unique(dataset['Time_point']).item()
    # save adata
    # adata.write_h5ad(
    #     '/lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/T_perturb/'
    #     'T_perturb/plt/res/Cora/cls_embeddings_stratified_pairing_.h5ad'
    # )

    sc.pp.neighbors(adata, use_rep='X_GF_zero_shot', n_neighbors=50)
    sc.tl.umap(adata)
    sc.pl.umap(
        adata,
        color=[
            'Cell_type',
            'Cell_population',
            'Cell_culture_batch',
            'Activation_level',
        ],
        wspace=0.5,
        ncols=2,
        frameon=False,
        show=False,
    )
    plt.savefig(
        f'./res/Geneformer/umap_cell_type_zeroshot_{time_point}.pdf',
        dpi=200,
        bbox_inches='tight',
    )
    plt.close()
