import os
import pickle
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from datasets import load_from_disk

# matplotlib style settings
from matplotlib import style
from tqdm import tqdm

style.use('default')
style.use(
    '/lustre/scratch123/hgi/projects/healthy_imm_expr/'
    't_generative/T_perturb/cytomeister/pp/mpl_style.mplstyle'
)

seed_no = 42
np.random.seed(seed_no)
if os.getcwd().split('/')[-3] != 'T_perturb':
    # set working directory to root of repository
    os.chdir(
        '/lustre/scratch123/hgi/projects/healthy_imm_expr/'
        't_generative/T_perturb/cytomeister/pp'
    )
    print('Changed working directory to root of repository')


# --- Explore tokenised data ---
# Filter adata for only DEGs
# read pickle file
with open(
    '/lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/'
    'T_perturb/tokenized_data/token_id_to_genename_hvg.pkl',
    'rb',
) as f:
    tokenid_to_hvg_genename = pickle.load(f)
unique_hvg_genes = list(tokenid_to_hvg_genename.values())
dataset = load_from_disk('./res/dataset/cytoimmgen_tokenised_hvg_paired.dataset')
# extract length of tokenised data
length = dataset['length']
# plot histogram of length
plt.hist(length, bins=100)
plt.xlabel('hvg/cell')
plt.ylabel('Counts')
plt.savefig('./res/tokenised_hvg/length_histogram.pdf', dpi=300, bbox_inches='tight')
plt.close()

# create pickle file with length
output_dir = './res/dataset'
with open(
    os.path.join(output_dir, 'cytoimmgen_tokenised_per_timepoint_length.pkl'), 'wb'
) as f:
    pickle.dump(length, f)

# load pkl file
with open(
    '/lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/'
    'generative_modelling_omic/Geneformer/geneformer/token_dictionary.pkl',
    'rb',
) as f:
    token_dictionary = pickle.load(f)
swapped_token_dictionary = {v: k for k, v in token_dictionary.items()}

input_ids_test = dataset[0]['input_ids']
# map ensembl ids to input_ids

ensembl_ids_list = []
for i in tqdm(range(len(dataset))):
    input_ids_tmp = dataset[i]['input_ids']
    ensembl_ids = [swapped_token_dictionary.get(i, None) for i in input_ids_tmp]
    ensembl_ids_list.append(ensembl_ids)
# load adata
adata = sc.read_h5ad('./res/h5ad_pairing_hvg/cytoimmgen_tokenised_hvg.h5ad')
# use adata var to map ensembl ids to gene names
ensembl_id_to_genename = dict(zip(adata.var_names, adata.var['gene_name']))
gene_name_list = [
    [ensembl_id_to_genename.get(i, None) for i in ensembl_ids]
    for ensembl_ids in ensembl_ids_list
]

# in gene_name_list if gene name is in unique_hvgs then append idx to dictionnary
hvg_idx_dict: Dict[str, list] = {}

for gene in tqdm(unique_hvg_genes):
    hvg_idx_dict[gene] = []
    for gene_name in gene_name_list:
        if gene in gene_name:
            # append index of gene
            hvg_idx_dict[gene].append(gene_name.index(gene))
        else:
            hvg_idx_dict[gene].append(np.nan)
# save dictionary
with open('./res/tokenised_hvg/hvgs_tokenisation_overlap.pkl', 'wb') as f:
    pickle.dump(hvg_idx_dict, f)
with open('./res/tokenised_hvg/hvgs_tokenisation_overlap.pkl', 'rb') as f:
    hvg_idx_dict = pickle.load(f)


# create dataframe
hvg_idx_df = pd.DataFrame.from_dict(hvg_idx_dict)
hvg_idx_df['Time_point'] = dataset['Time_point']
hvg_idx_df['Time_point'] = pd.Categorical(
    hvg_idx_df['Time_point'], ['0h', '16h', '40h', '5d']
)
# ignore nan values
hvg_idx_df[~hvg_idx_df['CD69'].isna()]['CD69'].plot(kind='hist', bins=100)
plt.xlabel('rank of hvg')
plt.ylabel('Counts')
plt.savefig(
    './res/tokenised_hvg/hvg_idx_histogram_CD69.pdf', dpi=300, bbox_inches='tight'
)
plt.close()
plt_CD69 = sns.violinplot(data=hvg_idx_df, y='CD69', hue='Time_point', orient='v')
plt_CD69.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
plt.xlabel('Timepoint')
plt.ylabel('Ranks')
plt.title('CD69')
plt.savefig('./res/tokenised_hvg/hvg_idx_violin_CD69.pdf', dpi=300, bbox_inches='tight')
plt.close()
hvg_idx_df[~hvg_idx_df['IL2RA'].isna()]['IL2RA'].plot(kind='hist', bins=100)
plt.xlabel('rank of hvg')
plt.ylabel('Counts')
plt.savefig(
    './res/tokenised_hvg/hvg_idx_histogram_IL2RA.pdf', dpi=300, bbox_inches='tight'
)
plt.close()
plt_IL2RA = sns.violinplot(data=hvg_idx_df, y='IL2RA', hue='Time_point', orient='v')
plt_IL2RA.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
plt.xlabel('Timepoint')
plt.ylabel('Ranks')
plt.title('IL2RA')
plt.savefig(
    './res/tokenised_hvg/hvg_idx_violin_IL2RA.pdf', dpi=300, bbox_inches='tight'
)
plt.close()
# plot violin plot of hvgs

plt_IL7R = sns.violinplot(data=hvg_idx_df, y='IL7R', hue='Time_point', orient='v')
plt_IL7R.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
plt.xlabel('Timepoint')
plt.ylabel('Ranks')
plt.title('IL7R')
plt.savefig('./res/tokenised_hvg/hvg_idx_violin_IL7R.pdf', dpi=300, bbox_inches='tight')
plt.close()
# check for columns with only nan values
nan_columns = hvg_idx_df.columns[hvg_idx_df.isna().all()].tolist()
print(f'Columns where all values are NaN: {nan_columns}')

mean_rank = hvg_idx_df.iloc[:, :-1].mean()
# calculate mean expression of hvgs
adata.var_names = adata.var['gene_name']
expression = adata[:, hvg_idx_df.iloc[:, :-1].columns].X.A

# calculate mean expression of hvgs
mean_expression = []
non_zero = []
for i in range(expression.shape[1]):
    # non-zero median expression
    tmp = expression[:, i]
    tmp_ = tmp[tmp != 0]
    mean_expression.append(np.mean(tmp_))
    # count number of non-zero values
    non_zero.append(len(tmp_))

mean_df = pd.DataFrame(
    {'Mean_rank': mean_rank, 'Mean_expression': mean_expression, 'Non_zero': non_zero}
)
# create spearman correltion scatter plot
# compute spearman correlation
pearson_corr = mean_df.corr(method='pearson')
fig = plt.figure()
ax = fig.add_subplot(111)
sc = ax.scatter(
    mean_df['Mean_rank'], mean_df['Mean_expression'], s=10, c=mean_df['Non_zero']
)
plt.colorbar(sc)
plt.xlabel('Mean rank')
plt.ylabel('Non-zero mean expression')
plt.savefig(
    './res/tokenised_hvg/mean_rank_vs_mean_expression.pdf',
    dpi=300,
    bbox_inches='tight',
)
plt.close()
