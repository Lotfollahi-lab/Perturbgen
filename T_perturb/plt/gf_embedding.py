import os

import anndata as ad
import pandas as pd
import scanpy as sc
import torch
from datasets import load_from_disk
from geneformer import EmbExtractor
from matplotlib import pyplot as plt
from matplotlib import style

# from transformers import BertForSequenceClassification, BertForTokenClassification

style.use(
    '/lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/'
    'T_perturb/T_perturb/pp/mpl_style.mplstyle'
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
    '/lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/'
    'T_perturb/T_perturb/pp/res/dataset/'
    'cytoimmgen_degs_random_pairing_16h.dataset'
)
dataset = load_from_disk(tokenized_dir)
num_labels = len(set(list(dataset['Cell_type'])))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


embex = EmbExtractor(
    model_type='CellClassifier',
    num_classes=num_labels,  # number of cell types for unsupervised training
    emb_mode='cell',
    max_ncells=len(dataset),  # extract embeddings for all cells
    emb_layer=0,  # 0 = to last layer, -1 = second to last layer
    forward_batch_size=32,
    emb_label=['Cell_population', 'Cell_type'],
    labels_to_plot=['Cell_population', 'Cell_type'],
    nproc=8,
    summary_stat=None,
)

# embs = embex.extract_embs(
#     './res/Geneformer/240131_geneformer_CellClassifier_L2048_B32_LR5e-05_LSlinear_WU10000_E3_Oadamw_F5_16h/checkpoint-16272',
#     '/lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/'
#     'T_perturb/T_perturb/pp/res/dataset/cytoimmgen_degs_random_pairing_16h.dataset',
#     './res/Geneformer',
#     'cell_embeddings_finetuned_16h',
# )
embs = embex.extract_embs(
    '/lustre/scratch123/hgi/projects/healthy_imm_expr/'
    't_generative/generative_modelling_omic/Geneformer',
    '/lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/'
    'T_perturb/T_perturb/pp/res/dataset/cytoimmgen_degs_random_pairing_16h.dataset',
    './res/Geneformer',
    'cell_embeddings_zeroshot_16h',
)
embs_df = pd.read_csv(
    '/lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/'
    'T_perturb/T_perturb/plt/res/Geneformer/cell_embeddings_zeroshot_16h.csv',
    index_col=0,
)

emb_label = ['Cell_type', 'Cell_population']
emb_dims = embs.shape[1] - len(emb_label)
only_embs_df = embs_df.iloc[:, :emb_dims]
only_embs_df.index = pd.RangeIndex(0, only_embs_df.shape[0], name=None).astype(str)
only_embs_df.columns = pd.RangeIndex(0, only_embs_df.shape[1], name=None).astype(str)
vars_dict = {'embs': only_embs_df.columns}

obs_dict = {
    'cell_id': list(only_embs_df.index),
    'Cell_type': list(embs_df[emb_label[0]]),
    'Cell_population': list(embs_df[emb_label[1]]),
}
adata = ad.AnnData(X=only_embs_df, obs=obs_dict, var=vars_dict)
adata_ = adata.copy()
adata_.obs['activation'] = None
adata_.obs['activation'][adata_.obs['Cell_population'].str.contains('LA')] = 'LA'
adata_.obs['activation'][~adata_.obs['Cell_population'].str.contains('LA')] = 'HA'
emb_label = emb_label + ['activation']
sc.tl.pca(adata_, svd_solver='arpack')
sc.pp.neighbors(adata_)
sc.tl.umap(adata_)
sc.pl.umap(adata_, color=emb_label, save=False, frameon=False, wspace=0.3)
plt.savefig(
    './res/Geneformer/umap_cell_type_zeroshot_16h.pdf', dpi=200, bbox_inches='tight'
)
plt.close()
