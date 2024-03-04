import os
import random

import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import seaborn as sns
from matplotlib import style

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
time_point = '16h'

# Plotting CLS embeddings
# --------------------------------

adata = sc.read_h5ad(
    f'/lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/T_perturb/'
    f'T_perturb/plt/res/Cora/'
    f'cls_embeddings_stratified_pairing_{time_point}_cosine_similarity.h5ad'
)

# Plotting log normalised embeddings
# --------------------------------

# plot log normalised embeddings
sc.tl.pca(adata, svd_solver='arpack', n_comps=50)
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=50)
sc.tl.umap(adata)
adata.obsm['X_lognorm_umap'] = adata.obsm['X_umap']
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
    basis='X_lognorm_umap',
    color=[
        'Cell_type',
        'Cell_population',
        'Cell_culture_batch',
        'Activation_level',
    ],
    ncols=2,
    wspace=0.3,
    frameon=False,
    show=False,
)
plt.savefig(f'./res/umap_lognorm_{time_point}.pdf', dpi=300, bbox_inches='tight')
plt.close()

# plot umap of cls embeddings
fig, ax = plt.subplots(figsize=(5, 5))
sc.pp.neighbors(adata, n_neighbors=15, use_rep='X_CLS_embeddings')
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
    basis='X_CLS_umap',
    color=[
        'Cell_type',
        'Cell_population',
        'Cell_culture_batch',
        'Activation_level',
    ],
    ncols=2,
    wspace=0.3,
    frameon=False,
    show=False,
)
plt.savefig(
    f'./res/Cora/cls_embeddings_umap_{time_point}.pdf',
    bbox_inches='tight',
)
plt.close()

adata = sc.read_h5ad(
    f'/lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/T_perturb/'
    f'T_perturb/plt/res/Cora/'
    f'cls_embeddings_stratified_pairing_{time_point}_cosine_similarity.h5ad'
)
var_names = adata.obsm['cosine_similarity'].columns
adata.var_names = adata.var['gene_name']
# filter adata to only include genes in var_names
adata = adata[:, var_names]
adata.obs['Cell_type_activity'] = (
    adata.obs['Cell_type'].astype(str) + '_' + adata.obs['Activation_level'].astype(str)
)
adata.layers['cosine_similarity'] = adata.obsm['cosine_similarity']
sc.pl.dotplot(
    adata,
    var_names,
    groupby='Cell_type_activity',
    dendrogram=False,
    layer='cosine_similarity',
    show=False,
    swap_axes=True,
    var_group_rotation=60,
)

# save figure
plt.savefig(
    f'./res/Cora/cosine_similarity_{time_point}.pdf',
    bbox_inches='tight',
)
plt.close()
