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

embs = embex.extract_embs(
    './res/Geneformer/'
    '240129_geneformer_CellClassifier_L2048_B32_LR5e-05_LSlinear_WU500_E4_Oadamw_F5/'
    'checkpoint-17380',
    '/lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/'
    'T_perturb/T_perturb/pp/res/dataset/cytoimmgen_degs_random_pairing_16h.dataset',
    './res/Geneformer',
    'cell_embeddings',
)

embs_df = pd.read_csv(
    '/lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/'
    'T_perturb/T_perturb/plt/res/Geneformer/cell_embeddings.csv',
    index_col=0,
)
emb_dims = 256
only_embs_df = embs_df.iloc[:, :emb_dims]
only_embs_df.index = pd.RangeIndex(0, only_embs_df.shape[0], name=None).astype(str)
only_embs_df.columns = pd.RangeIndex(0, only_embs_df.shape[1], name=None).astype(str)
vars_dict = {'embs': only_embs_df.columns}
label = ['Cell_type', 'Cell_population']
obs_dict = {
    'cell_id': list(only_embs_df.index),
    'Cell_type': list(dataset[label[0]]),
    'Cell_population': list(dataset[label[1]]),
}
adata = ad.AnnData(X=only_embs_df, obs=obs_dict, var=vars_dict)
sc.tl.pca(adata, svd_solver='arpack')
sc.pp.neighbors(adata)
sc.tl.umap(adata)
sc.pl.umap(adata, color=label, save=False)
plt.savefig('./res/Geneformer/umap_cell_type.pdf', dpi=300, bbox_inches='tight')
