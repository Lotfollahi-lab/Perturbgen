import scanpy as sc
import pickle
import pandas as pd
import numpy as np
import gc
import anndata as ad
from typing import Dict
from anndata import AnnData
from gears.inference import compute_metrics, deeper_analysis, non_dropout_analysis

model = '20240515_1502_ttransformer'
# model = '20240515_1006_ttransformer'
# model = '20240515_0736_ttransformer'
# model = '20240513_1715_ttransformer'
# model = '20240507_1018_ttransformer' # mse
base_path = '/lustre/scratch126/cellgen/team361/ip14/Projects/2024Mar_Tperturb'
data_path = 'datasets/Norman2019'
pp_path = 'T_perturb/T_perturb/pp/res'

mse_loss = True

def compute_perturbation_metrics(
    results: Dict,
    ctrl_adata: AnnData,
    non_zero_genes: bool = False,
    return_raw: bool = False,
) -> Dict:
    """
    Given results from a model run and the ground truth, compute metrics

    Args:
        results (:obj:`Dict`): The results from a model run
        ctrl_adata (:obj:`AnnData`): The adata of the control condtion
        non_zero_genes (:obj:`bool`, optional): Whether to only consider non-zero
            genes in the ground truth when computing metrics
        return_raw (:obj:`bool`, optional): Whether to return the raw metrics or
            the mean of the metrics. Default is False.

    Returns:
        :obj:`Dict`: The metrics computed
    """
    from scipy.stats import pearsonr

    # metrics:
    #   Pearson correlation of expression on all genes, on DE genes,
    #   Pearson correlation of expression change on all genes, on DE genes,

    metrics_across_genes = {
        "pearson": [],
        "pearson_de": [],
        "pearson_delta": [],
        "pearson_de_delta": [],
    }

    metrics_across_conditions = {
        "pearson": [],
        "pearson_delta": [],
    }

    conditions = np.unique(results["pert_cat"])
    assert not "ctrl" in conditions, "ctrl should not be in test conditions"
    condition2idx = {c: np.where(results["pert_cat"] == c)[0] for c in conditions}

    mean_ctrl = np.array(ctrl_adata.X.mean(0)).flatten()  # (n_genes,)
    # assert ctrl_adata.X.max() <= 1000, "gene expression should be log transformed"

    true_perturbed = results["truth"]  # (n_cells, n_genes)
    # assert true_perturbed.max() <= 1000, "gene expression should be log transformed"
    true_mean_perturbed_by_condition = np.array(
        [true_perturbed[condition2idx[c]].mean(0) for c in conditions]
    )  # (n_conditions, n_genes)
    true_mean_delta_by_condition = true_mean_perturbed_by_condition - mean_ctrl
    zero_rows = np.where(np.all(true_mean_perturbed_by_condition == 0, axis=1))[
        0
    ].tolist()
    zero_cols = np.where(np.all(true_mean_perturbed_by_condition == 0, axis=0))[
        0
    ].tolist()

    pred_perturbed = results["pred"]  # (n_cells, n_genes)
    pred_mean_perturbed_by_condition = np.array(
        [pred_perturbed[condition2idx[c]].mean(0) for c in conditions]
    )  # (n_conditions, n_genes)
    pred_mean_delta_by_condition = pred_mean_perturbed_by_condition - mean_ctrl

    def corr_over_genes(x, y, conditions, res_list, skip_rows=[], non_zero_mask=None):
        """compute pearson correlation over genes for each condition"""
        for i, c in enumerate(conditions):
            if i in skip_rows:
                continue
            x_, y_ = x[i], y[i]
            if non_zero_mask is not None:
                x_ = x_[non_zero_mask[i]]
                y_ = y_[non_zero_mask[i]]
            res_list.append(pearsonr(x_, y_)[0])

    corr_over_genes(
        true_mean_perturbed_by_condition,
        pred_mean_perturbed_by_condition,
        conditions,
        metrics_across_genes["pearson"],
        zero_rows,
        non_zero_mask=true_mean_perturbed_by_condition != 0 if non_zero_genes else None,
    )
    corr_over_genes(
        true_mean_delta_by_condition,
        pred_mean_delta_by_condition,
        conditions,
        metrics_across_genes["pearson_delta"],
        zero_rows,
        non_zero_mask=true_mean_perturbed_by_condition != 0 if non_zero_genes else None,
    )

    def find_DE_genes(adata, condition, geneid2idx, non_zero_genes=False, top_n=20):
        """
        Find the DE genes for a condition
        """
        '''
        key_components = next(
            iter(adata.uns["rank_genes_groups_cov_all"].keys())
        ).split("_")
        assert len(key_components) == 3, "rank_genes_groups_cov_all key is not valid"

        condition_key = "_".join([key_components[0], condition, key_components[2]])

        de_genes = adata.uns["rank_genes_groups_cov_all"][condition_key]
        if non_zero_genes:
            de_genes = adata.uns["top_non_dropout_de_20"][condition_key]
            # de_genes = adata.uns["rank_genes_groups_cov_all"][condition_key]
            # de_genes = de_genes[adata.uns["non_zeros_gene_idx"][condition_key]]
            # assert len(de_genes) > top_n
        '''
        de_genes = adata.uns["rank_genes_groups_cov_all"][condition]
        if non_zero_genes:
            de_genes = adata.uns["top_non_dropout_de_20"][condition]
            
        de_genes = de_genes[:top_n]

        de_idx = [geneid2idx[i] for i in de_genes]

        return de_idx, de_genes

    geneid2idx = dict(zip(ctrl_adata.var.index.values, range(len(ctrl_adata.var))))
    de_idx = {
        c: find_DE_genes(ctrl_adata, c, geneid2idx, non_zero_genes)[0]
        for c in conditions
    }
    mean_ctrl_de = np.array(
        [mean_ctrl[de_idx[c]] for c in conditions]
    )  # (n_conditions, n_diff_genes)

    true_mean_perturbed_by_condition_de = np.array(
        [
            true_mean_perturbed_by_condition[i, de_idx[c]]
            for i, c in enumerate(conditions)
        ]
    )  # (n_conditions, n_diff_genes)
    zero_rows_de = np.where(np.all(true_mean_perturbed_by_condition_de == 0, axis=1))[
        0
    ].tolist()
    true_mean_delta_by_condition_de = true_mean_perturbed_by_condition_de - mean_ctrl_de

    pred_mean_perturbed_by_condition_de = np.array(
        [
            pred_mean_perturbed_by_condition[i, de_idx[c]]
            for i, c in enumerate(conditions)
        ]
    )  # (n_conditions, n_diff_genes)
    pred_mean_delta_by_condition_de = pred_mean_perturbed_by_condition_de - mean_ctrl_de

    corr_over_genes(
        true_mean_perturbed_by_condition_de,
        pred_mean_perturbed_by_condition_de,
        conditions,
        metrics_across_genes["pearson_de"],
        zero_rows_de,
    )
    corr_over_genes(
        true_mean_delta_by_condition_de,
        pred_mean_delta_by_condition_de,
        conditions,
        metrics_across_genes["pearson_de_delta"],
        zero_rows_de,
    )

    if not return_raw:
        for k, v in metrics_across_genes.items():
            metrics_across_genes[k] = np.mean(v)
        for k, v in metrics_across_conditions.items():
            metrics_across_conditions[k] = np.mean(v)
    metrics = metrics_across_genes

    return metrics

adata = sc.read_h5ad(f'{base_path}/T_perturb/T_perturb/plt/res/Petra/{model}_generate_pred_adata_None.h5ad')
adata_ = sc.read_h5ad(f'{base_path}/{data_path}/adata/filtered_tokenised_hvg.h5ad')
adata.uns['top_non_dropout_de_20'] = adata_.uns['top_non_dropout_de_20']
adata.uns['top_non_zero_de_20'] = adata_.uns['top_non_zero_de_20']
adata.uns['non_dropout_gene_idx'] = adata_.uns['non_dropout_gene_idx']
adata.uns['non_zeros_gene_idx'] = adata_.uns['non_zeros_gene_idx']
adata.uns['rank_genes_groups_cov_all'] = adata_.uns['rank_genes_groups_cov_all']
adata.uns['perturbation_id'] = adata_.uns['perturbation_id']
adata.uns['perturbation_id'].index = adata.uns['perturbation_id'].index.astype(str)
adata.var = adata_.var.set_index('rowidx', drop=False)
adata.obs.index = adata.obs.tgt_cell_idx

# del adata_
gc.collect()

with open(f'{base_path}/{pp_path}/Petra/pert_test_split_seed1.pkl', 'rb') as f:
    subgroup = pickle.load(f)

all_perts = subgroup['test_subgroup']['combo_seen0'] + subgroup['test_subgroup']['combo_seen1'] + subgroup['test_subgroup']['combo_seen2'] + subgroup['test_subgroup']['unseen_single']
subgroup['test_subgroup']['all'] = all_perts

pert_idx = adata.obs['perturbation_id'].unique()
idx2gene = {}
for idx in pert_idx:
    if '+' in idx:
        idx2gene[idx] = '+'.join(adata.uns['perturbation_id'].loc[idx.split('+'),'gene_name'])
    else:
        idx2gene[idx] = adata.uns['perturbation_id'].loc[idx.split('+'),'gene_name'].iloc[0]+'+ctrl'

reverse = [pert for pert in idx2gene.values() if not pert in all_perts]
for idx, gene in idx2gene.items():
    if gene in reverse:
        idx2gene[idx] = '+'.join(adata.uns['perturbation_id'].loc[idx.split('+'),'gene_name'][::-1])
        
[pert for pert in idx2gene.values() if not pert in all_perts]
adata.obs['condition'] = adata.obs['perturbation_id'].map(idx2gene)

for pert in all_perts:
    if '+ctrl' in pert:
        adata.uns['top_non_dropout_de_20'][pert] = adata.uns['top_non_dropout_de_20'].pop(pert.replace('+ctrl',''))
        adata.uns['top_non_zero_de_20'][pert] = adata.uns['top_non_zero_de_20'].pop(pert.replace('+ctrl',''))
        adata.uns['rank_genes_groups_cov_all'][pert] = adata.uns['rank_genes_groups_cov_all'].pop(pert.replace('+ctrl',''))
        
adata.uns['non_zeros_gene_idx'] = {k.replace('A549_','').replace('_1+1',''): v for k, v in adata.uns['non_zeros_gene_idx'].items()}
adata.uns['non_dropout_gene_idx'] = {k.replace('A549_','').replace('_1+1',''): v for k, v in adata.uns['non_dropout_gene_idx'].items()}
adata.obs = adata.obs.drop(columns=['tgt_cell_idx','src_cell_idx','perturbation_id'])

results = {}
results['pert_cat'] = adata.obs['condition']
results["pred"] = adata.X
results["truth"] = adata.layers['tgt_counts']

results["pred_de"] = []
results["truth_de"] = []

for i, cell in enumerate(adata.obs.index):
    de_genes_idx = adata.uns['top_non_dropout_de_20'][adata.obs['condition'][cell]]
    results["pred_de"].append(adata.X[i,de_genes_idx])
    results["truth_de"].append(adata.layers['tgt_counts'][i,de_genes_idx])

results["pred_de"] = np.stack(results["pred_de"])
results["truth_de"] = np.stack(results["truth_de"])

adata.X = adata.layers['tgt_counts']
adata.obs['condition_name'] = adata.obs['condition']
del adata.layers['src_counts']
del adata.layers['tgt_counts']

adata_ = adata_[adata_.obs.condition == 'ctrl']
adata_.obs = pd.DataFrame(
    {
        'condition': 'ctrl',
        'condition_name': 'ctrl'
    }, index=adata_.obs.cell_pairing_index)

# if model in ['20240428_1333', '20240430_1104']: # normalize ctrls for mse-trained decoders
if mse_loss:
    sc.pp.normalize_total(adata_, target_sum=1e4)
    sc.pp.log1p(adata_)

adata_.uns = adata.uns
adata_.var = adata.var
del adata_.obsm
del adata_.varm
del adata_.obsp
del adata_.layers['counts']
adata = ad.concat([adata,adata_], join='outer', merge='first', uns_merge='first')
adata.var.index = adata.var.index.astype(int)

test_metrics, test_pert_res = compute_metrics(results)
print(test_metrics)

deeper_res = deeper_analysis(adata, results)
non_dropout_res = non_dropout_analysis(adata, results)

metrics = ["pearson_delta", "pearson_delta_de"]
metrics_non_dropout = [
    "pearson_delta_top20_de_non_dropout",
    "pearson_top20_de_non_dropout",
]
subgroup_analysis = {}
for name in subgroup["test_subgroup"].keys():
    subgroup_analysis[name] = {}
    for m in metrics:
        subgroup_analysis[name][m] = []

    for m in metrics_non_dropout:
        subgroup_analysis[name][m] = []

for name, pert_list in subgroup["test_subgroup"].items():
    for pert in pert_list:
        for m in metrics:
            subgroup_analysis[name][m].append(deeper_res[pert][m])

        for m in metrics_non_dropout:
            subgroup_analysis[name][m].append(non_dropout_res[pert][m])

for name, result in subgroup_analysis.items():
    for m in result.keys():
        subgroup_analysis[name][m] = np.mean(subgroup_analysis[name][m])

print(subgroup_analysis)
print(compute_perturbation_metrics(results, adata_))





