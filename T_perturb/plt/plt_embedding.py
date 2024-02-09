import os

import matplotlib.pyplot as plt
import scanpy as sc
import seaborn as sns
from matplotlib import style

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

adata = sc.read_h5ad(
    '/lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/'
    'T_perturb/T_perturb/plt/res/scConformer/cls_embeddings_5d.h5ad'
)
adata_ = adata.copy()
adata_.obs['activation'] = None
adata_.obs['activation'][adata_.obs['cell_population'].str.contains('LA')] = 'LA'
adata_.obs['activation'][~adata_.obs['cell_population'].str.contains('LA')] = 'HA'
# plot umap of cls embeddings
fig, ax = plt.subplots(figsize=(1, 1))
sc.pp.neighbors(adata_, n_neighbors=15, use_rep='X')
sc.tl.umap(adata_)
# use colorblind friendly palette
sc.pl.umap(
    adata_,
    color=[
        'cell_type',
        'cell_population',
        'activation',
    ],  # leave gap between cell type and cell population
    wspace=0.5,
    frameon=False,
    show=False,
)
plt.savefig(
    '/lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/'
    'T_perturb/T_perturb/pp/res/cls_embeddings_umap_5d.pdf',
    bbox_inches='tight',
)
plt.close()

adata = sc.read_h5ad(
    '/lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/T_perturb/'
    'T_perturb/pp/res/h5ad_data/cytoimmgen_tokenisation_degs_16h.h5ad'
)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=5000)
sc.tl.pca(adata, svd_solver='arpack', n_comps=50, use_highly_variable=True)
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=50)
sc.tl.umap(adata)
adata_ = adata.copy()
adata_.obs['activation'] = None
adata_.obs['activation'][adata_.obs['Cell_population'].str.contains('LA')] = 'LA'
adata_.obs['activation'][~adata_.obs['Cell_population'].str.contains('LA')] = 'HA'
sc.pl.umap(
    adata_,
    color=['Cell_type', 'Cell_population', 'activation'],
    save='umap_test',
    frameon=False,
    wspace=0.3,
)
plt.savefig('./res/umap_lognorm_16h.pdf', dpi=300, bbox_inches='tight')
plt.close()
