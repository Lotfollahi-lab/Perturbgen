import os
import sys

import anndata as ad
import pandas as pd
import scanpy as sc
import scgen
import torch

# from torchmetrics import PearsonCorrCoef

if os.getcwd().split('/')[-2] != 'T_perturb':
    # set working directory to root of repository
    os.chdir(
        '/lustre/scratch123/hgi/projects/healthy_imm_expr/' 't_generative/T_perturb/'
    )
    print('Changed working directory to root of repository')


from Model.metric import pearson
from src.utils import stratified_split

sys.path.append(
    '/lustre/scratch123/hgi/projects/healthy_imm_expr/'
    't_generative/T_perturb//benchmarking/scgen'
)
RANDOM_SEED = 42

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
# save pred
pred.write('./benchmarking/res/saved_models/pred_perturbation_prediction.h5ad')
# subset for only perturbed cells
pred_counts = torch.tensor(pred[test.obs['Time_point'] == '16h'].X)
test_counts = torch.tensor(test[test.obs['Time_point'] == '16h'].X.A)

pearson_mean = pearson(pred_counts, test_counts)
# pearson delta
ctrl = torch.tensor(adata_0h_test.X.A)
pearson_delta_mean = pearson(pred_counts, test_counts, ctrl)

# random baseline
adata_random = sc.pp.subsample(adata_full, n_obs=pred.shape[0], copy=True)
pred_random = torch.tensor(adata_random[test.obs['Time_point'] == '16h'].X.A)
random_pearson_mean = pearson(pred_random, test_counts)
random_pearson_delta_mean = pearson(pred_random, test_counts, ctrl)
results = pd.DataFrame(
    {
        'pearson': [pearson_mean.item(), random_pearson_mean.item()],
        'pearson_delta': [pearson_delta_mean.item(), random_pearson_delta_mean.item()],
        'model': ['scgen', 'random'],
    }
)
# save the results
results.to_csv('./benchmarking/res/scgen_random_reproducibility.csv', index=False)
