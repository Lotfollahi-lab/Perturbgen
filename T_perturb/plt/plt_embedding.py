import argparse
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from matplotlib import style

from T_perturb.Model.metric import (
    evaluate_emd,
    evaluate_mmd,
    lin_reg_summary,
)

np.random.seed(42)
random.seed(42)


if os.getcwd().split('/')[-1] != 'healthy_imm_expr':
    # set working directory to root of repository
    os.chdir('/lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/')
    print('Changed working directory to root of repository')


def get_args():
    """Get command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--res_dir',
        type=str,
        # default='./T_perturb/T_perturb/plt/res/eb',
        default='./T_perturb/T_perturb/plt/res/cytoimmgen',
        help='Dataset to use for analysis',
    )
    parser.add_argument(
        '--full_data_dir',
        type=str,
        # default='./T_perturb/T_perturb/pp/res/'
        # 'h5ad_pairing_hvg/cytoimmgen_tokenised_hvg.h5ad',
        default=(
            './T_perturb/T_perturb/pp/eb/res/'
            'h5ad_pairing_hvg/cytoimmgen_tokenised_hvg.h5ad'
        ),
        help='Dataset to use for analysis',
    )
    args = parser.parse_args()
    return args


args = get_args()
# colorblind friendly palette
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=sns.color_palette('colorblind'))

style.use('./T_perturb/T_perturb/pp/mpl_style.mplstyle')

# Plotting CLS embeddings
# --------------------------------


# Plotting log normalised embeddings
# --------------------------------

# plot log normalised embeddings
adata_full = sc.read_h5ad(args.full_data_dir)

sc.pp.normalize_total(adata_full, target_sum=1e4)
sc.pp.log1p(adata_full)
sc.tl.pca(adata_full, svd_solver='arpack', n_comps=50)
sc.pp.neighbors(adata_full, n_neighbors=15, n_pcs=50)
sc.tl.umap(adata_full)
adata_full.obsm['X_lognorm_umap'] = adata_full.obsm['X_umap']

sc.pl.embedding(
    adata_full,
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
adata_cls = sc.read_h5ad(
    f'{args.res_dir}/'
    '20240822_random_pairing_stratified_pairing'
    '_generate_adata_[1]_GF_fine_tuned_42_zinb_3.h5ad'
)
var_names = adata_cls.obsm['cosine_similarity'].columns
# filter adata to only include genes in var_names
adata_cls = adata_cls[:, var_names]
# create highly active and lowly active column
adata_cls.layers['cosine_similarity'] = adata_cls.obsm['cosine_similarity']
del adata_cls.obsm['cosine_similarity']
# categorise time points [16h, 40h, 5d]
adata_cls.obs['time_point'] = adata_cls.obs['time_point'].cat.reorder_categories(
    ['16h', '40h', '5d']
)
marker_genes = [
    'IL7R',
    'CD52',
    'LTB',
    'CXCR4',
    'STAT1',
    'IRF1',
    # 'IFIT3',
    'GBP1',
    # 'SYNE2',
    # 'SOCS3',
    'IL4R',
    'CD69',
    'TNFRSF4',
    'HSP90AA1',
    'FABP5',
    'TUBA1B',
    'IL2RA',
    'BATF',
    # 'CORO1B',
    'ISG15',
    # 'ALDOC',
    'DDIT4',
    # 'LGALS1',
    # 'S100A4',
    'CD74',
    # 'HLA-DRA',
    # 'HLA-DRB1',
]
# reorder genes based on marker genes
adata_cls.var
adata_cls = adata_cls[:, marker_genes]
fig, ax = plt.subplots(figsize=(6, 10))
sc.pl.dotplot(
    adata_cls,
    marker_genes,
    use_raw=False,
    groupby=['Time_point', 'Cell_type'],
    dendrogram=False,
    layer='cosine_similarity',
    show=False,
    swap_axes=True,
    var_group_rotation=60,
    ax=ax,
)
# save figure
plt.savefig(
    f'{args.res_dir}/cosine_similarity_all_cells.pdf',
    bbox_inches='tight',
)
plt.close()

# creating plot for lowly active T cells
adata_cls.obs['Activation_level'] = None
# if .obs['cell_population'] endswith LA then
# Activation_level = lowly active else highly active
adata_cls.obs.loc[
    adata_cls.obs['Cell_population'].str.endswith('LA'), 'Activation_level'
] = 'Lowly active'
adata_cls.obs.loc[
    ~adata_cls.obs['Cell_population'].str.endswith('LA'), 'Activation_level'
] = 'Highly active'
# only for 16h time point
adata_cls_16h = adata_cls[adata_cls.obs['Time_point'] == '16h']
# plot umap
sc.pp.neighbors(adata_cls_16h, n_neighbors=15, use_rep='cls_embeddings')
sc.tl.umap(adata_cls_16h)
sc.pl.embedding(
    adata_cls_16h,
    basis='X_umap',
    color=[
        'Activation_level',
    ],
    frameon=False,
    show=False,
)
plt.savefig(
    f'{args.res_dir}/cls_embeddings_umap_activation_lvl_16h_all_cells.pdf',
    bbox_inches='tight',
)
fig, ax = plt.subplots(figsize=(4, 10))
sc.pl.dotplot(
    adata_cls_16h,
    marker_genes,
    use_raw=False,
    groupby=['Activation_level'],
    dendrogram=False,
    layer='cosine_similarity',
    show=False,
    swap_axes=True,
    var_group_rotation=45,
    ax=ax,
)
# save figure
plt.savefig(
    f'{args.res_dir}/cosine_similarity_LA_HA_16h.pdf',
    bbox_inches='tight',
)
plt.close()
# plot umap of cls embeddings
fig, ax = plt.subplots(figsize=(5, 5))
# create umap for each time point separately
for time_point in adata_cls.obs['Time_point'].cat.categories:
    adata_time = adata_cls[adata_cls.obs['Time_point'] == time_point]
    sc.pp.neighbors(adata_time, n_neighbors=15, use_rep='cls_embeddings')
    sc.tl.umap(adata_time)
    sc.pl.embedding(
        adata_time,
        basis='X_umap',
        color=[
            'Cell_type',
            'Cell_population',
            'Activation_level',
            # 'batch',
        ],
        ncols=3,
        wspace=0.15,
        frameon=False,
        show=False,
    )
    plt.savefig(
        f'{args.res_dir}/cls_embeddings_umap_{time_point}_all_cells.pdf',
        bbox_inches='tight',
    )
    plt.close()

# full umap
sc.pp.neighbors(adata_cls, n_neighbors=15, use_rep='cls_embeddings')
sc.tl.umap(adata_cls)
sc.pl.embedding(
    adata_cls,
    basis='X_umap',
    color=[
        'Cell_type',
        'Cell_population',
        'Time_point',
        # 'batch',
        # 'Activation_level',
    ],
    ncols=2,
    wspace=0.3,
    frameon=False,
    show=False,
)
plt.savefig(
    f'{args.res_dir}/'
    '20240822_random_pairing_generate_'
    'reproducibillity_umap_generate_s100_full.pdf',
    bbox_inches='tight',
)
plt.close()
# sc.pp.neighbors(adata, n_neighbors=15, use_rep='cls_embeddings')
# sc.tl.umap(adata)
# adata.obsm['X_CLS_umap'] = adata.obsm['X_umap']
# # use colorblind friendly palette
# # sc.pl.umap(
# #     adata,
# #     color=[
# #         'Cell_type',
# #         'Cell_population',
# #         'Cell_culture_batch',
# #         'Activation_level',
# #     ],  # leave gap between cell type and cell population
# #     wspace=0.5,
# #     ncols=2,
# #     #plot 2x2 grid

# #     frameon=False,
# #     show=False,
# # )
# sc.pl.embedding(
#     adata,
#     basis='X_CLS_umap',
#     color=[
#         'cell_type',
#         'cell_population',
#         'time_point',
#         'batch',
#     ],
#     ncols=2,
#     wspace=0.3,
#     frameon=False,
#     show=False,
# )
# plt.savefig(
#     './res/Petra/full_data_cls_embeddings_umap.pdf',
#     bbox_inches='tight',
# )
# plt.close()

# plot gene embeddings
# --------------------------------
# plot gene embedding for each gene separately
for gene, i in adata_cls.uns['activation_genes'].items():
    print(i)
    # filter gene embeddings with non-zero values
    gene_embedding = adata_cls.obsm['gene_embeddings'][:, i, :]
    # filter embeddings with non-zero values
    non_zero_mask = np.all(gene_embedding != 0, axis=1)
    gene_embedding_non_zero = gene_embedding[non_zero_mask,]
    adata_non_zero = adata_cls[non_zero_mask,]
    adata_non_zero.obsm['gene_embeddings'] = gene_embedding_non_zero
    # print gene embedding umap
    sc.pp.neighbors(adata_non_zero, n_neighbors=15, use_rep='gene_embeddings')
    sc.tl.umap(adata_non_zero)
    sc.pl.embedding(
        adata_non_zero,
        basis='X_umap',
        color=[
            'Cell_type',
            'Cell_population',
            'Time_point',
            'Activation_level',
            'batch',
        ],
        frameon=False,
        show=False,
    )
    plt.savefig(
        f'{args.res_dir}/gene_embeddings_umap_{gene}.pdf',
        bbox_inches='tight',
    )
    plt.close()


# Plotting generate results
# --------------------------------
adata = sc.read_h5ad(
    f'{args.res_dir}/'
    '20240813_generate_adata_extrapolate_[3]__GF_fine_tuned_42_zinb_3.h5ad'
)
del adata.uns['Cell_type_colors']
del adata.uns['Cell_population_colors']
del adata.uns['Time_point_colors']
# plot embeddings
sc.pp.neighbors(adata, n_neighbors=15, use_rep='cls_embeddings')
sc.tl.umap(adata)
sc.pl.embedding(
    adata,
    basis='X_umap',
    color=[
        'Cell_type',
        'Cell_population',
        'Time_point',
    ],
    ncols=2,
    wspace=0.4,
    frameon=False,
    show=False,
)
plt.savefig(
    f'{args.res_dir}/generate_umap_cls.pdf',
    bbox_inches='tight',
)
# log normalised counts
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.tl.pca(adata, svd_solver='arpack', n_comps=50)
sc.pp.neighbors(adata, n_neighbors=15, n_pcs=50)
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
plt.savefig(f'{args.res_dir}/generate_umap_raw.pdf', dpi=300, bbox_inches='tight')
plt.close()

mode = 'log_norm'
# plot true anndata
adata_true = adata.copy()
adata_true.X = adata_true.layers['counts']
if mode == 'log_norm':
    sc.pp.normalize_total(adata_true, target_sum=1e4)
    sc.pp.log1p(adata_true)

sc.tl.pca(adata_true, svd_solver='arpack', n_comps=50)
sc.pp.neighbors(adata_true, n_neighbors=15, n_pcs=50)
sc.tl.umap(adata_true)
sc.pl.umap(
    adata_true,
    color=[
        'Cell_type',
        'Time_point',
        'Cell_population',
    ],
    wspace=0.5,
    # plot 2x2 grid
    frameon=False,
    show=False,
)
plt.savefig(f'./res/true_umap_{mode}.pdf', dpi=300, bbox_inches='tight')
adata_full = sc.read_h5ad(
    './T_perturb/T_perturb/pp/res/cytoimmgen/h5ad_pairing_hvg/cytoimmgen_hvg.h5ad'
)
adata_random = adata_full.copy()
sc.pp.subsample(adata_random, n_obs=adata_true.n_obs)
# calculate emd and mmd between true and generated data for cytoimmgen
emd_list = []
emd_random = []
mmd_list = []
mmd_random = []
for time_point in adata_true.obs['Time_point'].cat.categories:
    adata_time_true = adata_true[adata_true.obs['Time_point'] == time_point]
    adata_time_pred = adata[adata.obs['Time_point'] == time_point]
    emd_df = evaluate_emd(adata_time_true, adata_time_pred, 'Cell_type')
    emd_df['Time_point'] = time_point
    emd_list.append(emd_df)
    emd_random_df = evaluate_emd(adata_time_true, adata_random, 'Cell_type')
    emd_random_df['Time_point'] = time_point
    emd_random.append(emd_random_df)
    mmd_df = evaluate_mmd(adata_time_true, adata_time_pred, 'Cell_type')
    mmd_df['Time_point'] = time_point
    mmd_list.append(mmd_df)
    mmd_random_df = evaluate_mmd(adata_time_true, adata_random, 'Cell_type')
    mmd_random_df['Time_point'] = time_point
    mmd_random.append(mmd_random_df)

# create a dataframe to plot results as barplots the results are list of dictionnaries
emd_random_df = pd.concat(emd_random)
emd_random_df['Type'] = 'random'

emd_df = pd.concat(emd_list)
emd_df['Type'] = 'generated'
mmd_df = pd.concat(mmd_list)
mmd_df['Type'] = 'generated'
mmd_random_df = pd.concat(mmd_random)
mmd_random_df['Type'] = 'random'
df = pd.concat([emd_df, emd_random_df, mmd_df, mmd_random_df])
# create long df for plotting
df['Condition'] = df.index
df_long = pd.melt(
    df,
    id_vars=['Time_point', 'Type', 'Condition'],
    var_name='Metric',
    value_name='Value',
)
# drop nan values
df_long = df_long.dropna()
# plot emd and mmd per time point
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
sns.barplot(
    x='Time_point',
    y='Value',
    hue='Type',
    data=df_long[df_long['Metric'] == 'emd'],
    ax=ax[0],
    errorbar=None,
    legend=False,
)
ax[0].set_title('EMD')
ax[0].set_ylabel('EMD')
ax[0].set_xlabel('Time point')
sns.barplot(
    x='Time_point',
    y='Value',
    hue='Type',
    data=df_long[df_long['Metric'] == 'mmd'],
    ax=ax[1],
    errorbar=None,
)
ax[1].set_title('MMD')
ax[1].set_ylabel('MMD')
ax[1].set_xlabel('Time point')
plt.subplots_adjust(wspace=0.5)
plt.savefig('./res/emd_mmd_timepoint.pdf', bbox_inches='tight')
# save dataframe
df_long.to_csv('./res/emd_mmd_timepoint.csv')
df_long.groupby(['Metric', 'Type'])['Value'].mean()

# EB analysis
# ------------------------------
adata = sc.read_h5ad(
    f'{args.res_dir}/'
    'generate_adata_interpolate_encoder_ep_150_ckpt_19_Transformer_encoder_zinb_3.h5ad'
)
adata_true = adata.copy()
adata_true.X = adata_true.layers['counts']

emd_list = []
mmd_list = []
# print emd and mmd before normalisation
emd_df = evaluate_emd(adata_true, adata, None)
mmd_df = evaluate_mmd(adata=adata_true, pred_adata=adata, n_cells=10000)
print('EMD before normalisation: ', emd_df)
print('MMD before normalisation: ', mmd_df)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.normalize_total(adata_true, target_sum=1e4)
sc.pp.log1p(adata_true)
emd_df = evaluate_emd(adata_true, adata, None)
lin_reg_df = lin_reg_summary(adata_true, adata)

mmd_df = evaluate_mmd(adata=adata_true, pred_adata=adata, n_cells=10000)

print('EMD after normalisation: ', emd_df)
print('MMD after normalisation: ', mmd_df)
# concatenate results
metrics_df = pd.concat([emd_df, mmd_df, lin_reg_df], axis=1)
metrics_df.to_csv(f'{args.res_dir}/metrics_zinb_3_extrapolation_seed_42.csv')


# extrapolation of timepoints experiment
# -------------------------------------
adata_t2 = sc.read_h5ad(
    f'{args.res_dir}/generate_adata_extrapolate_t2_GF_frozen_zinb_3.h5ad'
)
adata_t3 = sc.read_h5ad(
    f'{args.res_dir}/generate_adata_extrapolate_t3_GF_frozen_zinb_3.h5ad'
)
adata_t4 = sc.read_h5ad(
    f'{args.res_dir}/generate_adata_extrapolate_t4_GF_frozen_zinb_3.h5ad'
)


def evaluate_extrapolation(adata_pred, time_point):
    adata_true = adata_pred.copy()
    adata_true.X = adata_true.layers['counts']
    """Evaluate the extrapolation of time points."""
    adata
    # sc.pp.normalize_total(adata_true, target_sum=1e4)
    # sc.pp.log1p(adata_true)
    # sc.pp.normalize_total(adata_pred, target_sum=1e4)
    # sc.pp.log1p(adata_pred)
    emd_df = evaluate_emd(adata_true, adata_pred)
    mmd_df = evaluate_mmd(adata_true, adata_pred)
    lin_reg_df = lin_reg_summary(adata_true, adata_pred)
    emd_df['Time_point'] = time_point
    mmd_df['Time_point'] = time_point
    lin_reg_df['Time_point'] = time_point
    return emd_df, mmd_df, lin_reg_df


emd_list = []
mmd_list = []
lin_reg_list = []

for time_point, adata_pred in zip([2, 3, 4], [adata_t2, adata_t3, adata_t4]):
    emd_df, mmd_df, lin_reg_df = evaluate_extrapolation(adata_pred, time_point)
    emd_list.append(emd_df)
    mmd_list.append(mmd_df)
    lin_reg_list.append(lin_reg_df)
# create barplot from emd and mmd
emd_df = pd.concat(emd_list)
mmd_df = pd.concat(mmd_list)
lin_reg_df = pd.concat(lin_reg_list)
# create one dataframe
df = pd.concat([emd_df, mmd_df, lin_reg_df], axis=1)
# delete duplicated columns
df = df.loc[:, ~df.columns.duplicated()]
# plot barplot with two metrics [mmd, rmse] and separate by time point
df_long = pd.melt(
    df,
    id_vars=['Time_point'],
    var_name='Metric',
    value_name='Value',
)

# create two separate plots one for RMSE and one for MMD
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
sns.lineplot(
    x='Time_point',
    y='Value',
    hue='Metric',
    data=df_long[df_long['Metric'] == 'mmd'],
    ax=ax[0],
    errorbar=None,
    legend=False,
    linewidth=2.0,
)
ax[0].set_ylabel('MMD')
ax[0].set_xlabel('Time point')
ax[0].locator_params(axis='y', nbins=4)
sns.lineplot(
    x='Time_point',
    y='Value',
    hue='Metric',
    data=df_long[df_long['Metric'] == 'pearson_r'],
    ax=ax[1],
    errorbar=None,
    legend=False,
    linewidth=2.0,
)
ax[1].set_ylabel('Pearson')
ax[1].set_xlabel('Time point')
# limit y axis range to 0.5 to 1
ax[1].set_ylim(0.7, 0.9)
# set max number of y ticks to 4
ax[1].locator_params(axis='y', nbins=4)
plt.subplots_adjust(wspace=0.5)

# set max number of y ticks to 4

plt.savefig(
    f'{args.res_dir}/extrapolation_mmd_pearson_timepoint.pdf', bbox_inches='tight'
)
plt.close()


# create two separate plots one for RMSE and one for MMD
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
sns.lineplot(
    x='Time_point',
    y='Value',
    hue='Metric',
    data=df_long[df_long['Metric'] == 'emd'],
    ax=ax[0],
    errorbar=None,
    legend=False,
    linewidth=2.0,
)
ax[0].set_ylabel('EMD')
ax[0].set_xlabel('Time point')
ax[0].locator_params(axis='y', nbins=4)
ax[0].set_ylim([0.1, 0.2])
sns.lineplot(
    x='Time_point',
    y='Value',
    hue='Metric',
    data=df_long[df_long['Metric'] == 'rmse'],
    ax=ax[1],
    errorbar=None,
    legend=False,
    linewidth=2.0,
)
ax[1].set_ylabel('RMSE')
ax[1].set_xlabel('Time point')
# limit y axis range to 0.5 to 1
# set max number of y ticks to 4
ax[1].locator_params(axis='y', nbins=4)
ax[1].set_ylim([0.25, 0.4])
plt.subplots_adjust(wspace=0.5)

# set max number of y ticks to 4

plt.savefig(f'{args.res_dir}/extrapolation_emd_rmse_timepoint.pdf', bbox_inches='tight')
plt.close()
