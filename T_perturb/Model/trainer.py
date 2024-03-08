import pickle
from typing import (
    Any,
    Dict,
    List,
    Optional,
)

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from geneformer.tokenizer import TOKEN_DICTIONARY_FILE
from pytorch_lightning import LightningModule
from torchmetrics import (
    CosineSimilarity,
    MeanSquaredError,
    SpearmanCorrCoef,
)
from torchmetrics.text import Perplexity

from T_perturb.Model.metric import (
    evaluate_emd,
    evaluate_mmd,
    pearson,
)
from T_perturb.Modules.T_model import (
    CountDecoder,
    Petra,
    cosine_schedule,
)
from T_perturb.src.losses import (
    mse_loss,
    nb,
    zinb,
)
from T_perturb.src.utils import one_hot_encoder

if torch.cuda.is_available():
    cuda_device_name = torch.cuda.get_device_name()
    # If the device is an A100, set the precision for matrix multiplication
    if 'A100' in cuda_device_name:
        torch.set_float32_matmul_precision('medium')

sc.settings.set_figure_params(dpi=500)


def batch_token_ranking(tokenised_cells: torch.tensor, vocab_size: int):
    batch_size, seq_length = tokenised_cells.shape

    # tensor of size (batch_size, seq_length) filled with 0 to seq_length
    indices = torch.arange(seq_length, device=tokenised_cells.device).expand(
        batch_size, -1
    )

    # tensor of size (batch_size, vocab_size) filled with seq_length
    first_occurrence_indices = torch.full(
        (batch_size, vocab_size),
        seq_length,
        dtype=torch.long,
        device=tokenised_cells.device,
    )

    # For each vocab item, find the first occurrence index
    for i in range(vocab_size):
        mask = tokenised_cells == i
        valid_indices = torch.where(mask, indices, seq_length)
        first_occurrence_indices[:, i] = valid_indices.min(dim=-1).values

    return first_occurrence_indices


class Petratrainer(LightningModule):
    def __init__(
        self,
        tgt_vocab_size: int = 25000,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 1,
        d_ff: int = 2048,
        max_seq_length: int = 2048,
        dropout: float = 0.0,
        mlm_probability: float = 0.15,
        weight_decay: float = 0.0,
        lr: float = 1e-3,
        lr_scheduler_patience: float = 5.0,
        return_embeddings: bool = False,
        generate: bool = False,
        batch_size: int = 32,
        dataset_info: Optional[str] = None,
        adata: Optional[ad.AnnData] = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.transformer = Petra(
            tgt_vocab_size=tgt_vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=d_ff,
            max_seq_length=max_seq_length,
            dropout=dropout,
            mlm_probability=mlm_probability,
        )
        self.target_device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.masking_loss = nn.CrossEntropyLoss()
        self.save_hyperparameters()
        self.weight_decay = weight_decay
        self.lr = lr
        self.lr_scheduler_patience = lr_scheduler_patience
        # self.lr_scheduler_factor = lr_scheduler_factor
        self.metric = nn.ModuleDict(
            {
                'perplexity': Perplexity(ignore_index=-100),
                'cosine_similarity': CosineSimilarity(reduction='mean'),
                'mse': MeanSquaredError(),
                # 'rmse': MeanSquaredError(squared=False),
                'spearman': SpearmanCorrCoef(num_outputs=batch_size),
            }
        )

        with open(TOKEN_DICTIONARY_FILE, 'rb') as f:
            gene_token_dict = pickle.load(f)
        self.gene_token_dict = {value: key for key, value in gene_token_dict.items()}
        with open(
            '/lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/T_perturb/'
            'T_perturb/pp/res/token_dictionary_degs_for_subset_token_id.pkl',
            'rb',
        ) as f:
            self.subset_tokenid_to_deg = pickle.load(f)
        self.return_embeddings = return_embeddings
        self.generate = generate
        self.cls_embeddings_list: List[torch.tensor] = []
        self.gene_embeddings_list: List[torch.tensor] = []
        self.token_id_list: List[torch.tensor] = []
        self.tgt_vocab_size = tgt_vocab_size
        self.adata = adata
        self.dataset_info = dataset_info

    def forward(self, batch):
        outputs = self.transformer(
            src_input_id=batch['src_input_ids'],
            tgt_input_id_t1=batch['tgt_input_ids_t1'],
            tgt_input_id_t2=batch['tgt_input_ids_t2'],
            original_lens=batch['src_length'],
            generate=self.generate,
        )
        return outputs

    def configure_optimizers(self):
        parameters = [{'params': self.transformer.parameters(), 'lr': self.lr}]
        optimizer = optim.Adam(parameters, weight_decay=self.weight_decay)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=self.lr_scheduler_patience,
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': lr_scheduler,
            'monitor': 'train/loss',
        }

    def training_step(self, batch, *args, **kwargs):
        # logits, labels, count_output, count_dropout = self.forward(batch)
        outputs = self.forward(batch)
        logits = outputs['logits']
        labels = outputs['labels']

        perp = Perplexity(ignore_index=-100).to('cuda')  # -100 = masked labels
        perp.update(logits, labels)
        logits = logits.contiguous().view(-1, logits.size(-1))
        labels = labels.contiguous().view(-1)

        masking_loss = self.masking_loss(logits, labels)

        self.log(
            'train/loss',
            masking_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch['tgt_input_ids_t1'].shape[0],
        )

        self.log(
            'train/perplexity',
            perp.compute(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch['tgt_input_ids_t1'].shape[0],
        )
        return masking_loss

    def on_train_epoch_end(self):
        # return F1 score and accuracy
        pass

    def validation_step(self, batch, *args, **kwargs):
        outputs = self.forward(batch)
        logits = outputs['logits']
        labels = outputs['labels']
        perp = Perplexity(ignore_index=-100).to('cuda')
        perp.update(logits, labels)
        logits = logits.contiguous().view(-1, logits.size(-1))
        labels = labels.contiguous().view(-1)
        masking_loss = self.masking_loss(logits, labels)
        self.log(
            'val/loss',
            masking_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch['tgt_input_ids_t1'].shape[0],
        )
        self.log(
            'val/perplexity',
            perp.compute(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch['tgt_input_ids_t1'].shape[0],
        )

    def test_step(self, batch, *args, **kwargs):
        if self.return_embeddings:
            outputs = self.forward(batch)
            self.cls_embeddings_list.append(outputs['dec_embedding'][:, 0, :])
            self.gene_embeddings_list.append(outputs['dec_embedding'][:, 1:, :])
            self.token_id_list.append(batch['tgt_input_ids'])
            # self.tgt_output['cell_type'].append(batch['tgt_cell_type'])
            # self.tgt_output['cell_population'].append(batch['tgt_cell_population'])
            # self.time_point = batch['tgt_time_point']

    def on_test_epoch_end(self):
        # return F1 score and accuracy
        if self.return_embeddings:
            print('Start saving embeddings -------------------')
            self.cls_embeddings_list = torch.cat(self.cls_embeddings_list)
            self.gene_embeddings_list = torch.cat(self.gene_embeddings_list)
            self.token_id_list = torch.cat(self.token_id_list)
            cosine_similarity_list = []
            for i in range(self.gene_embeddings_list.shape[0]):
                # gene level cosine similarity
                tmp_consine_similarity = F.cosine_similarity(
                    self.cls_embeddings_list[i],
                    self.gene_embeddings_list[i, :, :],
                    dim=1,
                )
                cosine_similarity_list.append(tmp_consine_similarity)
            cosine_similarity_list = torch.stack(cosine_similarity_list)

            # marker_genes = [
            #     'IL7R', 'CD52', 'GIMAP7', 'SARAF', 'BTG1',
            #     'LTB', 'CXCR4', 'STAT1', 'IRF1', 'IFIT3',
            #     'GBP1', 'SYNE2', 'SOCS3', 'IL4R', 'CD69',
            #     'MIR155HG', 'DDX21', 'TNFRSF4', 'HSP90AA1', 'HSP90AB1',
            #     'HSPA8', 'TXN', 'FABP5', 'TUBA1B', 'HMGA1',
            #     'PCNA', 'IL2RA', 'BATF', 'CD63'
            # ]

            marker_genes = [
                'IL7R',
                'CD52',
                'GIMAP7',
                'SARAF',
                'BTG1',
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
                'MIR155HG',
                'DDX21',
                'TNFRSF4',
                'HSP90AA1',
                'HSP90AB1',
                'HSPA8',
                'TXN',
                'FABP5',
                'TUBA1B',
                'HMGA1',
                'PCNA',
                'IL2RA',
                'BATF',
                'CD63',
                'IFITM2',
                'CORO1B',
                'ISG15',
                'ALDOC',
                'DDIT4',
                'LGALS1',
                'S100A4',
                'S100A6',
                'VIM',
                'CD74',
                'HLA-DRA',
                'HLA-DRB1',
            ]
            # filter for marker genes and swap key value
            marker_genes_ids = {
                v: k for k, v in self.subset_tokenid_to_deg.items() if v in marker_genes
            }

            emb = torch.zeros(
                cosine_similarity_list.shape[0], len(marker_genes_ids.keys())
            )
            for i, gene in enumerate(marker_genes_ids.keys()):
                cond_embs_to_fill = (self.token_id_list == marker_genes_ids[gene]).sum(
                    1
                ) > 0
                cond_select_markers = torch.where(
                    self.token_id_list == marker_genes_ids[gene]
                )
                cond_embs_to_fill = cond_embs_to_fill.cpu()
                emb[cond_embs_to_fill, i] = cosine_similarity_list[
                    cond_select_markers[0], cond_select_markers[1]
                ].cpu()
                # self.adata.obsm[marker_genes[i]] = emb.numpy()
            # create a dataframe and annotate columns as marker genes
            df = pd.DataFrame(emb.numpy(), columns=marker_genes_ids.keys())
            df.index = self.adata.obs_names
            self.adata.obsm['cosine_similarity'] = df

            self.cls_embeddings_list = self.cls_embeddings_list.detach().cpu().numpy()
            # save under adata.obsm
            self.adata.obsm['X_CLS_embeddings'] = self.cls_embeddings_list
            # save anndata
            self.adata.write_h5ad(
                f'/lustre/scratch123/hgi/projects/healthy_imm_expr/'
                f't_generative/T_perturb/T_perturb/'
                f'plt/res/Petra/'
                f'cls_embeddings_{self.dataset_info}_cosine_similarity.h5ad'
            )
            print('End saving embeddings -------------------')


class CountDecodertrainer(LightningModule):
    def __init__(
        self,
        ckpt_path: str = 'ckpt_path',
        loss_mode: str = 'mse',
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        lr_scheduler_patience: float = 1.0,
        # lr_scheduler_factor: float = 0.8,
        conditions: Optional[Dict[Any, Any]] = None,
        conditions_combined: Optional[List[Any]] = None,
        tgt_vocab_size: int = 25000,
        dropout: float = 0.0,
        d_model: int = 256,
        generate: bool = True,
        tgt_adata: Optional[ad.AnnData] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.target_device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        # Create an instance of your model
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        self.tgt_vocab_size = checkpoint['hyper_parameters']['tgt_vocab_size']
        self.d_model = checkpoint['hyper_parameters']['d_model']
        pretrained_model = Petra(
            tgt_vocab_size=self.tgt_vocab_size,
            d_model=self.d_model,
            d_ff=checkpoint['hyper_parameters']['d_ff'],
            max_seq_length=checkpoint['hyper_parameters']['max_seq_length'],
        )
        state_dict = checkpoint['state_dict']

        new_state_dict = {}
        for k, v in state_dict.items():
            k = k.replace('transformer.', '')
            new_state_dict[k] = v
        state_dict = new_state_dict

        pretrained_model.load_state_dict(new_state_dict, strict=False)

        self.decoder = CountDecoder(
            pretrained_model=pretrained_model,
            loss_mode=loss_mode,
            tgt_vocab_size=self.tgt_vocab_size,
            d_model=self.d_model,
            dropout=dropout,
        )
        self.save_hyperparameters()
        self.weight_decay = weight_decay
        self.lr = lr
        self.lr_scheduler_patience = lr_scheduler_patience
        # self.lr_scheduler_factor = lr_scheduler_factor
        self.loss_mode = loss_mode

        if (
            (self.loss_mode in ['nb', 'zinb'])
            and (conditions is not None)
            and (conditions_combined is not None)
        ):
            self.n_conditions = [len(conditions[cond]) for cond in conditions.keys()]
            self.conditions = conditions
            self.condition_encodings = {
                cond: {
                    k: v for k, v in zip(conditions[cond], range(len(conditions[cond])))
                }
                for cond in conditions.keys()
            }
            self.conditions_combined = conditions_combined
            self.n_conditions_combined = len(conditions_combined)
            self.conditions_combined_encodings = {
                k: v
                for k, v in zip(conditions_combined, range(len(conditions_combined)))
            }
            self.theta = torch.nn.Parameter(
                torch.randn(tgt_vocab_size - 1, self.n_conditions_combined)
            )
        else:
            self.theta = None

        self.metric = nn.ModuleDict(
            {
                'mse': MeanSquaredError(),
            }
        )
        self.generate = generate
        self.adata = tgt_adata
        # initiate lists to store true, ctrl and pred counts
        self.train_true_counts_list: List[int] = []
        self.train_pred_counts_list: List[int] = []
        self.val_true_counts_list: List[int] = []
        self.val_ctrl_counts_list: List[int] = []
        self.val_pred_counts_list: List[int] = []
        self.val_tgt_cell_type_list: List[str] = []
        self.val_tgt_cell_population_list: List[str] = []
        self.val_tgt_donor_list: List[str] = []
        self.test_true_counts_list: List[int] = []
        self.test_ctrl_counts_list: List[int] = []
        self.test_pred_counts_list: List[int] = []
        self.test_tgt_cell_type_list: List[str] = []
        self.test_tgt_cell_population_list: List[str] = []
        self.test_tgt_donor_list: List[str] = []

    def forward(self, batch):
        outputs = self.decoder(
            src_input_id=batch['src_input_ids'],
            tgt_input_id=batch['tgt_input_ids'],
            original_lens=batch['src_length'],
        )

        return outputs

    def compute_count_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ):
        true_counts = batch['tgt_counts'].float()
        batch_size_factor = np.array(batch['size_factor'])
        batch_size_factor = torch.tensor(batch_size_factor)
        batch_size_factor = batch_size_factor.to(self.target_device)

        if self.loss_mode == 'mse':
            loss = (
                mse_loss(outputs['count_lognorm'], true_counts)
                .sum(dim=-1)
                .mean()
                .float()
            )
            return loss, outputs['count_lognorm']

        elif self.loss_mode == 'zinb':
            combined_batch = torch.tensor(batch['combined_batch'])
            combined_batch = combined_batch.to(self.target_device)
            dec_mean_gamma, dec_dropout = (
                outputs['count_mean'],
                outputs['count_dropout'],
            )
            size_factor_view = batch_size_factor.unsqueeze(1).expand(
                dec_mean_gamma.size(0), dec_mean_gamma.size(1)
            )

            dec_mean = dec_mean_gamma * size_factor_view

            dispersion = F.linear(
                one_hot_encoder(combined_batch, self.n_conditions_combined), self.theta
            )
            dispersion = torch.exp(dispersion)
            loss = (
                -zinb(x=true_counts, mu=dec_mean, theta=dispersion, pi=dec_dropout)
                .sum(dim=-1)
                .mean()
            )
            return loss, dec_mean

        elif self.loss_mode == 'nb':
            combined_batch = torch.tensor(batch['combined_batch'])
            combined_batch = combined_batch.to(self.target_device)
            dec_mean_gamma = outputs['count_mean']

            size_factor_view = batch_size_factor.unsqueeze(1).expand(
                dec_mean_gamma.size(0), dec_mean_gamma.size(1)
            )
            dec_mean = dec_mean_gamma * size_factor_view
            dispersion = F.linear(
                one_hot_encoder(combined_batch, self.n_conditions_combined), self.theta
            )
            dispersion = torch.exp(dispersion)
            loss = -nb(x=true_counts, mu=dec_mean, theta=dispersion).sum(dim=-1).mean()
            return loss, dec_mean

        else:
            raise ValueError('Loss not supported, choose either mse or zinb')

    def training_step(self, batch, *args, **kwargs):
        outputs = self.forward(batch)
        count_loss, pred_count = self.compute_count_loss(outputs, batch)

        self.log(
            'train/loss',
            count_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch['tgt_input_ids'].shape[0],
        )
        # MSE
        mse = self.metric['mse'](pred_count, batch['tgt_counts'])
        mean_mse = torch.mean(mse)
        self.log(
            'train/mse',
            mean_mse,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        # implement the split
        # pearson delta

        # pearson top20 deg
        # random

        # against random

        # RMSE
        # rmse=self.metric['rmse'](outputs['count_mean'], batch['tgt_counts'])
        # mean_rmse = torch.mean(rmse)
        # self.log(
        #     'train/rmse',
        #     mean_rmse,
        #     on_epoch=True,
        #     prog_bar=True,
        #     logger=True,
        # )
        self.train_true_counts_list.append(batch['tgt_counts'])
        self.train_pred_counts_list.append(pred_count)

        return count_loss

    def on_train_epoch_end(self):
        # return Pearson correlation coefficient
        true_counts = torch.cat(self.train_true_counts_list).detach().cpu()
        pred_counts = torch.cat(self.train_pred_counts_list).detach().cpu()
        # Pearson correlation coefficient
        mean_pearson = pearson(pred_counts=pred_counts, true_counts=true_counts)
        self.log(
            'train/pearson',
            mean_pearson,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        # set to status quo
        self.train_true_counts_list = []
        self.train_pred_counts_list = []

    def validation_step(self, batch, *args, **kwargs):
        outputs = self.forward(batch)
        count_loss, pred_count = self.compute_count_loss(outputs, batch)
        self.log(
            'val/loss',
            count_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch['tgt_input_ids'].shape[0],
        )
        # MSE
        mse = self.metric['mse'](pred_count, batch['tgt_counts'])
        mean_mse = torch.mean(mse)
        self.log(
            'val/mse',
            mean_mse,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.val_true_counts_list.append(batch['tgt_counts'])
        self.val_pred_counts_list.append(pred_count)
        self.val_ctrl_counts_list.append(batch['src_counts'])
        self.val_tgt_cell_type_list.append(batch['tgt_cell_type'])
        self.val_tgt_cell_population_list.append(batch['tgt_cell_population'])
        self.val_tgt_donor_list.append(batch['tgt_donor'])
        return count_loss

    def on_validation_epoch_end(self):
        # return Pearson correlation coefficient
        true_counts = torch.cat(self.val_true_counts_list).detach().cpu()
        pred_counts = torch.cat(self.val_pred_counts_list).detach().cpu()
        ctrl_counts = torch.cat(self.val_ctrl_counts_list).detach().cpu()
        # tgt_cell_type = np.concatenate(self.val_tgt_cell_type_list)
        # tgt_cell_population = np.concatenate(self.val_tgt_cell_population_list)
        # tgt_donor = np.concatenate(self.val_tgt_donor_list)
        # val_obs = pd.DataFrame(
        #     np.array([tgt_cell_type, tgt_cell_population, tgt_donor]).T,
        #     columns=['Cell_type', 'Cell_population', 'Donor'],
        # )
        # pred_adata = ad.AnnData(
        #     X=pred_counts.numpy(),
        #     obs=val_obs,
        #     var=self.adata.var
        #     )
        # Pearson correlation coefficient
        mean_pearson = pearson(pred_counts=pred_counts, true_counts=true_counts)
        self.log(
            'val/pearson',
            mean_pearson,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        mean_pearson_delta = pearson(
            pred_counts=pred_counts, true_counts=true_counts, ctrl_counts=ctrl_counts
        )
        self.log(
            'val/pearson_delta',
            mean_pearson_delta,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        # mmd = evaluate_mmd(self.adata, pred_adata, condition_key='Cell_type')
        # print('MMD:', mmd)

        # emd = evaluate_emd(self.adata, pred_adata, condition_key='Cell_type')
        # print('EMD: ', emd)
        # set to status quo
        self.val_true_counts_list = []
        self.val_ctrl_counts_list = []
        self.val_pred_counts_list = []
        self.val_tgt_cell_type_list = []
        self.val_tgt_cell_population_list = []
        self.val_tgt_donor_list = []

    def test_step(self, batch, *args, **kwargs):
        if self.generate:
            outputs = self.decoder.generate(
                src_input_id=batch['src_input_ids'],
                noise_schedule=cosine_schedule,
                tgt_input_id=batch['tgt_input_ids'],
                original_lens=batch['src_length'],
                can_remask_prev_masked=False,
                topk_filter_thres=0.9,
                temperature=2.0,
                timesteps=18,
            )
            count_loss, pred_count = self.compute_count_loss(outputs, batch)
            self.log(
                'test/loss',
                count_loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                batch_size=batch['tgt_input_ids'].shape[0],
            )
        else:
            outputs = self.forward(batch)
            count_loss, pred_count = self.compute_count_loss(outputs, batch)
            self.log(
                'test/loss',
                count_loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                batch_size=batch['tgt_input_ids'].shape[0],
            )

            self.test_pred_counts_list.append(pred_count)
            self.test_true_counts_list.append(batch['tgt_counts'])
            print(len(self.test_true_counts_list))
            self.test_ctrl_counts_list.append(batch['src_counts'])
            self.test_tgt_cell_type_list.append(batch['tgt_cell_type'])
            self.test_tgt_cell_population_list.append(batch['tgt_cell_population'])
            self.test_tgt_donor_list.append(batch['tgt_donor'])

    def on_test_epoch_end(self):
        # return Pearson correlation coefficient
        true_counts = torch.cat(self.test_true_counts_list).detach().cpu()
        pred_counts = torch.cat(self.test_pred_counts_list).detach().cpu()
        ctrl_counts = torch.cat(self.test_ctrl_counts_list).detach().cpu()
        tgt_cell_type = np.concatenate(self.test_tgt_cell_type_list)
        tgt_cell_population = np.concatenate(self.test_tgt_cell_population_list)
        tgt_donor = np.concatenate(self.test_tgt_donor_list)
        test_obs = pd.DataFrame(
            np.array([tgt_cell_type, tgt_cell_population, tgt_donor]).T,
            columns=['Cell_type', 'Cell_population', 'Donor'],
        )
        print(test_obs)
        pred_adata = ad.AnnData(X=pred_counts.numpy(), obs=test_obs, var=self.adata.var)
        pred_adata.layers['counts'] = true_counts.numpy()
        # save adata
        pred_adata.write_h5ad(
            f'/lustre/scratch123/hgi/projects/healthy_imm_expr/'
            f't_generative/T_perturb/T_perturb/'
            f'plt/res/Petra/'
            f'pred_adata_{self.dataset_info}.h5ad'
        )

        print(pred_adata)
        mean_pearson = pearson(pred_counts, true_counts)
        # Pearson correlation coefficient
        self.log(
            'test/pearson',
            mean_pearson,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        # Pearson delta
        mean_pearson_delta = pearson(pred_counts, true_counts, ctrl_counts)
        self.log(
            'test/pearson_delta',
            mean_pearson_delta,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        # MSE
        mse = self.metric['mse'](pred_counts, true_counts)
        mean_mse = torch.mean(mse)
        self.log(
            'test/mse',
            mean_mse,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        metrics = pd.DataFrame(
            {
                'pearson': [mean_pearson.cpu().detach().numpy()],
                'pearson_delta': [mean_pearson_delta.cpu().detach().numpy()],
                'mse': [mean_mse.cpu().detach().numpy()],
            }
        )
        metrics.to_csv(
            '/lustre/scratch123/hgi/projects/healthy_imm_expr/'
            't_generative/T_perturb/T_perturb/plt/res/Petra/'
            'test_count_metrics.csv'
        )
        # calculate MMD and EMD
        condition_key = 'Cell_population'
        mmd = evaluate_mmd(self.adata, pred_adata, condition_key=condition_key)
        mmd['metric'] = 'mmd'
        # rename column called mmd
        mmd = mmd.rename(columns={'mmd': 'value'})
        emd = evaluate_emd(self.adata, pred_adata, condition_key=condition_key)
        emd['metric'] = 'emd'
        emd = emd.rename(columns={'emd': 'value'})
        # concatenate
        metrics = pd.concat([mmd, emd])
        # save metrics
        metrics.to_csv(
            f'/lustre/scratch123/hgi/projects/healthy_imm_expr/'
            f't_generative/T_perturb/T_perturb/plt/res/Petra/'
            f'test_mmd_emd_{condition_key}_metrics.csv'
        )
        # set to status quo
        self.test_true_counts_list = []
        self.test_ctrl_counts_list = []
        self.test_pred_counts_list = []

    def configure_optimizers(self):
        parameters = [{'params': self.decoder.parameters(), 'lr': self.lr}]
        optimizer = optim.Adam(parameters, weight_decay=self.weight_decay)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=self.lr_scheduler_patience,
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': lr_scheduler,
            'monitor': 'train/loss',
        }
