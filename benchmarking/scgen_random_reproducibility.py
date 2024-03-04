import os
import sys

import anndata as ad
import pandas as pd
import scanpy as sc
import scgen
import torch
from torchmetrics import PearsonCorrCoef

if os.getcwd().split('/')[-1] != 'T_perturb':
    # set working directory to root of repository
    os.chdir(
        '/lustre/scratch123/hgi/projects/healthy_imm_expr/' 't_generative/T_perturb/'
    )
    print('Changed working directory to root of repository')


from T_perturb.src.utils import stratified_split

sys.path.append(
    '/lustre/scratch123/hgi/projects/healthy_imm_expr/'
    't_generative/T_perturb//benchmarking/scgen'
)
RANDOM_SEED = 42
adata = sc.read_h5ad(
    '/lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/T_perturb/'
    'T_perturb/pp/res/h5ad_pairing/'
    'cytoimmgen_tokenised_degs.h5ad'
)
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
adata_0h = sc.read_h5ad(
    '/lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/T_perturb/'
    'T_perturb/pp/res/h5ad_pairing/'
    'cytoimmgen_tokenisation_degs_stratified_pairing_0h.h5ad'
)
adata_16h = sc.read_h5ad(
    '/lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/T_perturb/'
    'T_perturb/pp/res/h5ad_pairing/'
    'cytoimmgen_tokenisation_degs_stratified_pairing_16h.h5ad'
)

train_indices, _, test_indices = stratified_split(
    tgt_adata=adata_16h,
    train_prop=0.8,
    test_prop=0.1,
    groups=['Cell_type', 'Donor'],
    seed=RANDOM_SEED,
)
# concatenate adata for log normalization
adata_full = ad.concat([adata_0h, adata_16h])
sc.pp.normalize_total(adata_full)
sc.pp.log1p(adata_full)
# split adata
adata_0h = adata_full[adata_full.obs['Time_point'] == '0h']
adata_16h = adata_full[adata_full.obs['Time_point'] == '16h']

# create train test split based on obs in adata
adata_0h.obs['train_test_split'] = None
adata_16h.obs['train_test_split'] = None
adata_0h_train = adata_0h.copy()[train_indices]
adata_0h_train.obs['train_test_split'] = 'train'
adata_16h_train = adata_16h.copy()[train_indices]
adata_16h_train.obs['train_test_split'] = 'train'
adata_0h_test = adata_0h.copy()[test_indices]
adata_0h_test.obs['train_test_split'] = 'test'
adata_16h_test = adata_16h.copy()[test_indices]
adata_16h_test.obs['train_test_split'] = 'test'

# concatenate train and test adata
train = ad.concat([adata_0h_train, adata_16h_train])
test = ad.concat([adata_0h_test, adata_16h_test])
scgen.SCGEN.setup_anndata(  # type: ignore
    train, batch_key='Time_point', labels_key='Cell_type'
)
model = scgen.SCGEN(train)  # type: ignore
model.save(
    './benchmarking/res/saved_models/model_perturbation_prediction.pt', overwrite=True
)
model.train(
    max_epochs=100, batch_size=32, early_stopping=True, early_stopping_patience=25
)
pred, delta = model.predict(ctrl_key='0h', stim_key='16h', adata_to_predict=test)


# compute pearson correlation
def compute_pearson(pred, test, time_point):
    pred_activated = pred[test.obs['Time_point'] == time_point]
    pred_activated_counts = torch.tensor(pred_activated.X)
    test_activated = test[test.obs['Time_point'] == time_point]
    test_activated_counts = torch.tensor(test_activated.X.A)
    pearson = PearsonCorrCoef(num_outputs=pred_activated.shape[0])
    pearson = pearson(pred_activated_counts.T, test_activated_counts.T)
    pearson_non_nan = pearson[~torch.isnan(pearson)]
    pearson_mean = torch.mean(pearson_non_nan)
    return pearson_mean


def compute_pearson_delta(pred, test, time_point, ctrl):
    pred_delta_act = pred[pred.obs['Time_point'] == time_point].X - ctrl
    pred_delta_act = torch.tensor(pred_delta_act)
    test_delta_act = test[test.obs['Time_point'] == time_point].X.A - ctrl
    test_delta_act = torch.tensor(test_delta_act)
    pearson_delta = PearsonCorrCoef(num_outputs=pred_delta_act.shape[0])
    pearson_delta = pearson_delta(pred_delta_act.T, test_delta_act.T)
    pearson_delta_na = pearson_delta[~torch.isnan(pearson_delta)]
    pearson_delta_mean = torch.mean(pearson_delta_na)
    return pearson_delta_mean


# subset for only perturbed cells

pearson_mean = compute_pearson(pred, test, '16h')
# pearson delta
ctrl = torch.tensor(adata_0h.X.A)
pearson_delta_mean = compute_pearson_delta(pred, test, '16h', ctrl)

# random baseline
adata_random = sc.pp.subsample(adata, n_obs=pred.shape[0], copy=True)
random_pearson_mean = compute_pearson(adata_random, test, '16h')
random_pearson_delta_mean = compute_pearson_delta(adata_random, test, '16h', ctrl)
results = pd.DataFrame(
    {
        'pearson': [pearson_mean.item(), random_pearson_mean.item()],
        'pearson_delta': [pearson_delta_mean.item(), random_pearson_delta_mean.item()],
        'model': ['scgen', 'random'],
    }
)
# save the results
results.to_csv('./benchmarking/res/scgen_random_reproducibility.csv', index=False)
