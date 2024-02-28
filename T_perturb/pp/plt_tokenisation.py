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
    't_generative/T_perturb/T_perturb/pp/mpl_style.mplstyle'
)

seed_no = 42
np.random.seed(seed_no)

# --- Explore tokenised data ---
# Filter adata for only DEGs
degs = pd.read_csv(
    '/lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/'
    'generative_modelling_omic_notebooks/'
    'pp/res/deg/significant_deg_1.5logfc_0.05padj_hvg_5k.csv'
)
unique_degs = degs['names'].unique()
dataset = load_from_disk('./res/dataset/cytoimmgen_tokenised_degs.dataset')
# extract length of tokenised data
length = dataset['length']
# plot histogram of length
plt.hist(length, bins=100)
plt.xlabel('DEG/cell')
plt.ylabel('Counts')
plt.savefig('./res/tokenised_deg/length_histogram.pdf', dpi=300, bbox_inches='tight')
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
adata = sc.read_h5ad('./res/h5ad_data/cytoimmgen_tokenisation_degs.h5ad')
# use adata var to map ensembl ids to gene names
ensembl_id_to_genename = dict(zip(adata.var_names, adata.var['gene_name']))
gene_name_list = [
    [ensembl_id_to_genename.get(i, None) for i in ensembl_ids]
    for ensembl_ids in ensembl_ids_list
]

# in gene_name_list if gene name is in unique_degs then append idx to dictionnary
degs_idx_dict: Dict[str, list] = {}

for gene in tqdm(unique_degs):
    degs_idx_dict[gene] = []
    for gene_name in gene_name_list:
        if gene in gene_name:
            # append index of gene
            degs_idx_dict[gene].append(gene_name.index(gene))
        else:
            degs_idx_dict[gene].append(np.nan)
# save dictionary
with open('./res/tokenised_deg/degs_tokenisation_overlap.pkl', 'wb') as f:
    pickle.dump(degs_idx_dict, f)
# create dataframe
degs_idx_df = pd.DataFrame.from_dict(degs_idx_dict)
degs_idx_df['Time_point'] = dataset['Time_point']
degs_idx_df['Time_point'] = pd.Categorical(
    degs_idx_df['Time_point'], ['0h', '16h', '40h', '5d']
)
# ignore nan values
degs_idx_df[~degs_idx_df['CD69'].isna()]['CD69'].plot(kind='hist', bins=100)
plt.xlabel('rank of degs')
plt.ylabel('Counts')
plt.savefig(
    './res/tokenised_deg/deg_idx_histogram_CD69.pdf', dpi=300, bbox_inches='tight'
)
plt.close()
sns.violinplot(data=degs_idx_df, y='CD69', hue='Time_point', orient='v')
plt.xlabel('Timepoint')
plt.ylabel('Ranks')
plt.title('CD69')
plt.savefig('./res/tokenised_deg/deg_idx_violin_CD69.pdf', dpi=300, bbox_inches='tight')
plt.close()
degs_idx_df[~degs_idx_df['IL2RA'].isna()]['IL2RA'].plot(kind='hist', bins=100)
plt.xlabel('rank of degs')
plt.ylabel('Counts')
plt.title('IL2RA')
plt.savefig(
    './res/tokenised_deg/deg_idx_histogram_IL2RA.pdf', dpi=300, bbox_inches='tight'
)
plt.close()
# plot violin plot of degs
sns.violinplot(data=degs_idx_df, y='IL2RA', hue='Time_point', orient='v')
plt.xlabel('Timepoint')
plt.ylabel('Ranks')
plt.title('IL2RA')
plt.savefig(
    './res/tokenised_deg/deg_idx_violin_IL2RA.pdf', dpi=300, bbox_inches='tight'
)
plt.close()
sns.violinplot(data=degs_idx_df, y='IL7R', hue='Time_point', orient='v')
plt.xlabel('Timepoint')
plt.ylabel('Ranks')
plt.title('IL7R')
plt.savefig('./res/tokenised_deg/deg_idx_violin_IL7R.pdf', dpi=300, bbox_inches='tight')
plt.close()
# check for columns with only nan values
nan_columns = degs_idx_df.columns[degs_idx_df.isna().all()].tolist()
print(f'Columns where all values are NaN: {nan_columns}')
