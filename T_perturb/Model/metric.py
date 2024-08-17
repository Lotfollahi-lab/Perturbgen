from typing import (
    Any,
    Dict,
    Optional,
)

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from scipy import stats
from scipy.sparse import issparse
from scipy.stats import wasserstein_distance
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import rbf_kernel


def pairwise_distance(x, y):
    x = x.view(x.shape[0], x.shape[1], 1)
    y = torch.transpose(y, 0, 1)
    output = torch.sum((x - y) ** 2, 1)
    output = torch.transpose(output, 0, 1)
    return output


def gaussian_kernel_matrix(x, y, alphas):
    """Computes multiscale-RBF kernel between x and y.
    Parameters
    ----------
    x: torch.Tensor
         Tensor with shape [batch_size, z_dim].
    y: torch.Tensor
         Tensor with shape [batch_size, z_dim].
    alphas: Tensor
    Returns
    -------
    Returns the computed multiscale-RBF kernel between x and y.
    """

    dist = pairwise_distance(x, y).contiguous()
    dist_ = dist.view(1, -1)

    alphas = alphas.view(alphas.shape[0], 1)
    beta = 1.0 / (2.0 * alphas)

    s = torch.matmul(beta, dist_)

    return torch.sum(torch.exp(-s), 0).view_as(dist)


def mmd_loss_calc(source_features, target_features, gamma):
    """Initializes Maximum Mean Discrepancy(MMD)
    between source_features and target_features.
    - Gretton, Arthur, et al. "A Kernel Two-Sample Test". 2012.
    Parameters
    ----------
    source_features: torch.Tensor
         Tensor with shape [batch_size, z_dim]
    target_features: torch.Tensor
         Tensor with shape [batch_size, z_dim]
    Returns
    -------
    Returns the computed MMD between x and y.
    """
    # alphas = [
    #     1e-6,
    #     1e-5,
    #     1e-4,
    #     1e-3,
    #     1e-2,
    #     1e-1,
    #     1,
    #     5,
    #     10,
    #     15,
    #     20,
    #     25,
    #     30,
    #     35,
    #     100,
    #     1e3,
    #     1e4,
    #     1e5,
    #     1e6,
    # ]
    # alphas = torch.autograd.Variable(torch.FloatTensor(alphas)).to(
    #     device=source_features.device
    # )

    # cost = torch.mean(
    #     gaussian_kernel_matrix(source_features, source_features, alphas)
    #     )
    # cost += torch.mean(
    #     gaussian_kernel_matrix(target_features, target_features, alphas)
    #     )
    # cost -= 2 * torch.mean(
    #     gaussian_kernel_matrix(source_features, target_features, alphas)
    # )
    xx = rbf_kernel(source_features, source_features, gamma)
    xy = rbf_kernel(source_features, target_features, gamma)
    yy = rbf_kernel(target_features, target_features, gamma)

    return xx.mean() + yy.mean() - 2 * xy.mean()


# Metrics below were adapted CellOT and CPA from:
# https://github.com/facebookresearch/CPA/blob/main/cpa/helper.py
# Date of access: 2024.01.08


def evaluate_mmd(
    adata,
    pred_adata,
    condition_key=None,
    de_genes_dict=None,
    n_cells=None,
):
    mmd_list = []
    if n_cells:
        if n_cells < adata.shape[0]:
            sc.pp.subsample(pred_adata, n_obs=n_cells)
            sc.pp.subsample(adata, n_obs=n_cells)
    if condition_key is not None:
        for cond in pred_adata.obs[condition_key].unique():
            adata_ = adata[adata.obs[condition_key] == cond].copy()
            pred_adata_ = pred_adata[pred_adata.obs[condition_key] == cond].copy()
            if issparse(adata_.X):
                adata_.X = adata_.X.A
            if issparse(pred_adata_.X):
                pred_adata_.X = pred_adata_.X.A

            gammas = [2, 1, 0.5, 0.1, 0.01, 0.005]
            print('start mmd calculation')
            mmd = np.mean(
                list(map(lambda x: mmd_loss_calc(adata_.X, pred_adata_.X, x), gammas))
            )
            print('end mmd calculation')
            # mmd = mmd_loss_calc(torch.Tensor(), torch.Tensor())
            mmd_list.append({'condition': cond, 'mmd': mmd})
            if de_genes_dict:
                de_genes = de_genes_dict[cond]
                sub_adata_ = adata_[:, de_genes]
                sub_pred_adata_ = pred_adata_[:, de_genes]
                mmd_deg = mmd_loss_calc(
                    torch.Tensor(sub_adata_.X), torch.Tensor(sub_pred_adata_.X)
                )
                mmd_list[-1]['mmd_deg'] = mmd_deg
        mmd_df = pd.DataFrame(mmd_list).set_index('condition')
    else:
        adata_ = adata.copy()
        pred_adata_ = pred_adata.copy()
        if issparse(adata_.X):
            adata_.X = adata_.X.A
        if issparse(pred_adata_.X):
            pred_adata_.X = pred_adata_.X.A
        gammas = [2, 1, 0.5, 0.1, 0.01, 0.005]
        mmd = np.mean(
            list(map(lambda x: mmd_loss_calc(adata_.X, pred_adata_.X, x), gammas))
        )
        mmd_list.append({'mmd': mmd})
        if de_genes_dict:
            de_genes = de_genes_dict
            sub_adata_ = adata_[:, de_genes]
            sub_pred_adata_ = pred_adata_[:, de_genes]
            mmd_deg = mmd_loss_calc(
                torch.Tensor(sub_adata_.X), torch.Tensor(sub_pred_adata_.X)
            )
            mmd_list[-1]['mmd_deg'] = mmd_deg
        mmd_df = pd.DataFrame(mmd_list)
    return mmd_df


def evaluate_emd(
    true_data: np.ndarray, pred_data: np.ndarray, condition_key=None, de_genes_dict=None
):
    emd_list = []
    if condition_key:  # instead of condition have it per timepoint
        for cond in pred_data.obs[condition_key].unique():
            adata_ = true_data[true_data.obs[condition_key] == cond].copy()
            pred_adata_ = pred_data[pred_data.obs[condition_key] == cond].copy()
            if issparse(adata_.X):
                adata_.X = adata_.X.A
            if issparse(pred_adata_.X):
                pred_adata_.X = pred_adata_.X.A
            wd = []
            for i, _ in enumerate(adata_.var_names):
                wd.append(
                    wasserstein_distance(
                        torch.Tensor(adata_.X[:, i]), torch.Tensor(pred_adata_.X[:, i])
                    )
                )
            emd_list.append({'condition': cond, 'emd': np.mean(wd)})
            if de_genes_dict:
                de_genes = de_genes_dict[cond]
                sub_adata_ = adata_[:, de_genes]
                sub_pred_adata_ = pred_adata_[:, de_genes]
                wd_deg = []
                for i, _ in enumerate(sub_adata_.var_names):
                    wd_deg.append(
                        wasserstein_distance(
                            torch.Tensor(sub_adata_.X[:, i]),
                            torch.Tensor(sub_pred_adata_.X[:, i]),
                        )
                    )
                emd_list[-1]['emd_deg'] = np.mean(wd_deg)
        emd_df = pd.DataFrame(emd_list).set_index('condition')

    else:
        true_data_ = true_data.copy()
        pred_data_ = pred_data.copy()
        wd = []
        for i, _ in enumerate(true_data_.var_names):
            if issparse(true_data_.X):
                true_data_.X = true_data_.X.A
            if issparse(pred_data_.X):
                pred_data_.X = pred_data_.X.A
            wd.append(
                wasserstein_distance(
                    torch.Tensor(true_data_.X[:, i]), torch.Tensor(pred_data_.X[:, i])
                )
            )
        emd_list.append({'emd': np.mean(wd)})
        emd_df = pd.DataFrame(emd_list)
    return emd_df


def lin_reg_summary(
    true_adata: sc.AnnData,
    pred_adata: sc.AnnData,
    condition_key: Optional[str] = None,
    de_genes_dict: Optional[Dict[Any, Any]] = None,
):
    if condition_key is not None:
        lin_reg_list = []
        for cond in pred_adata.obs[condition_key].unique():
            adata_ = true_adata[true_adata.obs[condition_key] == cond].copy()
            pred_adata_ = pred_adata[pred_adata.obs[condition_key] == cond].copy()
            if issparse(adata_.X):
                adata_.X = adata_.X.A
            if issparse(pred_adata_.X):
                pred_adata_.X = pred_adata_.X.A
            if de_genes_dict:
                de_genes = de_genes_dict[cond]
                adata_ = adata_[:, de_genes]
                pred_adata_ = pred_adata_[:, de_genes]
            x_true = np.average(adata_.X, axis=0)
            x_pred = np.average(pred_adata.X, axis=0)
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                x_true, x_pred
            )
            pearson_r = r_value**2
            lin_reg_list.append(
                {
                    'condition': cond,
                    'slope': slope,
                    'intercept': intercept,
                    'r_value': r_value,
                    'p_value': p_value,
                    'std_err': std_err,
                    'pearson_r': pearson_r,
                }
            )
        lin_reg_df = pd.DataFrame(lin_reg_list).set_index('condition')
    else:
        x_true = np.average(true_adata.X, axis=0)
        x_pred = np.average(pred_adata.X, axis=0)
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_true, x_pred)
        pearson_r = r_value**2
        rmse = mean_squared_error(x_true, x_pred, squared=False)
        lin_reg_df = pd.DataFrame(
            {
                'slope': [slope],
                'intercept': [intercept],
                'r_value': [r_value],
                'p_value': [p_value],
                'std_err': [std_err],
                'pearson_r': [pearson_r],
                'rmse': [rmse],
            },
            index=[0],
        )
        return lin_reg_df
