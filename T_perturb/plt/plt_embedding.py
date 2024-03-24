import os
import random

import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import seaborn as sns
from matplotlib import style

from T_perturb.Model.metric import evaluate_emd, evaluate_mmd

np.random.seed(42)
random.seed(42)

# colorblind friendly palette
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=sns.color_palette('colorblind'))
if os.getcwd().split('/')[-3] != 'T_perturb':
    # set working directory to root of repository
    os.chdir(
        '/lustre/scratch123/hgi/projects/healthy_imm_expr/'
        't_generative/T_perturb/T_perturb/plt'
    )
    print('Changed working directory to root of repository')

style.use(
    '/lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/'
    'T_perturb/T_perturb/pp/mpl_style.mplstyle'
)

# Plotting CLS embeddings
# --------------------------------

adata = sc.read_h5ad('./res/Petra/cls_embeddings_cosine_similarity.h5ad')

# Plotting log normalised embeddings
# --------------------------------

# plot log normalised embeddings
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.tl.pca(adata, svd_solver='arpack', n_comps=50)
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=50)
sc.tl.umap(adata)
adata.obsm['X_lognorm_umap'] = adata.obsm['X_umap']

sc.pl.embedding(
    adata,
    basis='X_lognorm_umap',
    color=[
        'cell_type',
        'cell_population',
        'time_point',
        'batch',
    ],
    ncols=2,
    wspace=0.3,
    frameon=False,
    show=False,
)
plt.savefig('./res/full_data_umap_log_norm.pdf', dpi=300, bbox_inches='tight')
plt.close()

# plot umap of cls embeddings
fig, ax = plt.subplots(figsize=(5, 5))
sc.pp.neighbors(adata, n_neighbors=15, use_rep='cls_embeddings')
sc.tl.umap(adata)
adata.obsm['X_CLS_umap'] = adata.obsm['X_umap']
# use colorblind friendly palette
# sc.pl.umap(
#     adata,
#     color=[
#         'Cell_type',
#         'Cell_population',
#         'Cell_culture_batch',
#         'Activation_level',
#     ],  # leave gap between cell type and cell population
#     wspace=0.5,
#     ncols=2,
#     #plot 2x2 grid

#     frameon=False,
#     show=False,
# )
sc.pl.embedding(
    adata,
    basis='cls_embeddings',
    color=[
        'cell_type',
        'cell_population',
        'time_point',
        'batch',
    ],
    ncols=2,
    wspace=0.3,
    frameon=False,
    show=False,
)
plt.savefig(
    './res/Petra/full_data_cls_embeddings_umap.pdf',
    bbox_inches='tight',
)
plt.close()


var_names = adata.obsm['cosine_similarity'].columns
# filter adata to only include genes in var_names
adata = adata[:, var_names]
adata.obs['Cell_type_activity'] = (
    adata.obs['cell_type'].astype(str) + '_' + adata.obs['Activation_level'].astype(str)
)
adata.layers['cosine_similarity'] = adata.obsm['cosine_similarity']
# categorise time points [16h, 40h, 5d]
adata.obs['time_point'] = adata.obs['time_point'].cat.reorder_categories(
    ['16h', '40h', '5d']
)
marker_genes = [
    'IL7R',
    'CD52',
    'LTB',
    'CXCR4',
    'STAT1',
    'IRF1',
    'IFIT3',
    'GBP1',
    'SYNE2',
    'SOCS3',
    'IL4R',
    'CD69',
    'DDX21',
    'TNFRSF4',
    'HSP90AA1',
    'HSP90AB1',
    'FABP5',
    'TUBA1B',
    'PCNA',
    'IL2RA',
    'BATF',
    'CORO1B',
    'ISG15',
    'ALDOC',
    'DDIT4',
    'LGALS1',
    'S100A4',
    'CD74',
    'HLA-DRA',
    'HLA-DRB1',
]
# reorder genes based on marker genes
adata.var
adata = adata[:, marker_genes]
sc.pl.dotplot(
    adata,
    marker_genes,
    use_raw=False,
    groupby=['time_point', 'cell_type'],
    dendrogram=False,
    layer='cosine_similarity',
    show=False,
    swap_axes=True,
    var_group_rotation=60,
)

# save figure
plt.savefig(
    './res/Petra/cosine_similarity.pdf',
    bbox_inches='tight',
)
plt.close()


# Plotting generate results
# --------------------------------
adata = sc.read_h5ad(
    '/lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/'
    'T_perturb/T_perturb/plt/res/Petra/generate_adata.h5ad'
)
# log normalised counts
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.tl.pca(adata, svd_solver='arpack', n_comps=50)
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=50)
sc.tl.umap(adata)
sc.pl.umap(
    adata,
    color=[
        'Cell_type',
        'Cell_population',
        'Time_point',
    ],
    wspace=0.5,
    # plot 2x2 grid
    frameon=False,
    show=False,
)
plt.savefig('./res/generate_umap_raw.pdf', dpi=300, bbox_inches='tight')
plt.close()

mode = 'log_norm'
# plot true anndata
adata_true = adata.copy()
adata_true.X = adata_true.layers['counts']
if mode == 'log_norm':
    sc.pp.normalize_total(adata_true, target_sum=1e4)
    sc.pp.log1p(adata_true)

sc.tl.pca(adata_true, svd_solver='arpack', n_comps=50)
sc.pp.neighbors(adata_true, n_neighbors=10, n_pcs=50)
sc.tl.umap(adata_true)
sc.pl.umap(
    adata_true,
    color=[
        'Cell_type',
        'Cell_population',
        'Time_point',
    ],
    wspace=0.5,
    # plot 2x2 grid
    frameon=False,
    show=False,
)
plt.savefig(f'./res/true_umap_{mode}.pdf', dpi=300, bbox_inches='tight')

# calculate emd and mmd between true and generated data
emd = evaluate_emd(adata_true, adata, 'Time_point')
mmd = evaluate_mmd(adata_true, adata, 'Time_point')
print(f'EMD: {emd}')
print(f'MMD: {mmd}')
# get random baseline
permutation = np.random.permutation(adata_true.n_obs)
adata_random = adata_true[permutation].copy()
emd_random = evaluate_emd(adata_true, adata_random, 'Time_point')
