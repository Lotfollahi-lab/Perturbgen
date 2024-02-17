import pickle
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Union,
)

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from geneformer.tokenizer import TOKEN_DICTIONARY_FILE
from pytorch_lightning import LightningModule
from torchmetrics import (
    CosineSimilarity,
    MeanSquaredError,
    SpearmanCorrCoef,
)
from torchmetrics.text import Perplexity

from T_perturb.Modules.T_model import TTransformer, cosine_schedule
from T_perturb.src.losses import (
    mse_loss,
    nb,
    zinb,
)
from T_perturb.src.utils import one_hot_encoder

wandb.init(
    entity='k-ly',
    project='ttransformer',
    dir='/lustre/scratch123/hgi/projects/healthy_imm_expr/'
    't_generative/T_perturb/T_perturb',
)
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


class TTransformertrainer(LightningModule):
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
        lr_scheduler_patience: float = 1.0,
        lr_scheduler_factor: float = 0.8,
        return_cls_embedding: bool = False,
        generate: bool = True,
        loss_mode: Union[str, str, str] = 'mse',
        alpha: float = 0.5,
        conditions: Optional[Dict[Any, Any]] = None,
        conditions_combined: Optional[List[Any]] = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.transformer = TTransformer(
            tgt_vocab_size=tgt_vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=d_ff,
            max_seq_length=max_seq_length,
            dropout=dropout,
            mlm_probability=mlm_probability,
            loss_mode=loss_mode,
        )
        self.target_device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.masking_loss = nn.CrossEntropyLoss()
        self.save_hyperparameters()
        self.weight_decay = weight_decay
        self.lr = lr
        self.lr_scheduler_patience = lr_scheduler_patience
        self.lr_scheduler_factor = lr_scheduler_factor
        self.metric = nn.ModuleDict(
            {
                'perplexity': Perplexity(ignore_index=-100),
                'cosine_similarity': CosineSimilarity(reduction='mean'),
                'mse': MeanSquaredError(),
                'spearman': SpearmanCorrCoef(num_outputs=64),
            }
        )
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
        self.alpha = alpha
        with open(TOKEN_DICTIONARY_FILE, 'rb') as f:
            gene_token_dict = pickle.load(f)
        self.gene_token_dict = {value: key for key, value in gene_token_dict.items()}
        self.return_cls_embedding = return_cls_embedding
        self.generate = generate
        self.cls_embeddings_list: List[torch.tensor] = []
        self.tgt_output: Dict[str, List[str]] = {'cell_type': [], 'cell_population': []}
        self.time_point = None
        self.tgt_vocab_size = tgt_vocab_size

    def compute_count_loss(self, outputs: torch.Tensor, batch: Dict[str, torch.Tensor]):
        true_counts = np.array(batch['tgt_counts'])
        true_counts = torch.tensor(true_counts).squeeze(1).float()
        true_counts = true_counts.to(self.target_device)
        batch_size_factor = np.array(batch['size_factor'])
        batch_size_factor = torch.tensor(batch_size_factor)
        batch_size_factor = batch_size_factor.to(self.target_device)

        if self.loss_mode == 'mse':
            pred_counts = outputs
            pred_counts = pred_counts[:, 1:]  # ignore CLS tokens
            loss = mse_loss(pred_counts, true_counts).sum(dim=-1).mean().float()
            return loss
        elif self.loss_mode == 'zinb':
            combined_batch = torch.tensor(batch['combined_batch'])
            combined_batch = combined_batch.to(self.target_device)
            dec_mean_gamma, dec_dropout = outputs
            dec_mean_gamma, dec_dropout = dec_mean_gamma[:, 1:], dec_dropout[:, 1:]
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

            return loss
        elif self.loss_mode == 'nb':
            combined_batch = torch.tensor(batch['combined_batch'])
            combined_batch = combined_batch.to(self.target_device)
            dec_mean_gamma = outputs
            dec_mean_gamma = dec_mean_gamma[:, 1:]
            size_factor_view = batch_size_factor.unsqueeze(1).expand(
                dec_mean_gamma.size(0), dec_mean_gamma.size(1)
            )
            dec_mean = dec_mean_gamma * size_factor_view
            dispersion = F.linear(
                one_hot_encoder(combined_batch, self.n_conditions_combined), self.theta
            )
            dispersion = torch.exp(dispersion)
            loss = -nb(x=true_counts, mu=dec_mean, theta=dispersion).sum(dim=-1).mean()
            return loss

        else:
            raise ValueError('Loss not supported, choose either mse or zinb')

    def forward(self, batch):
        if self.training:
            logits, labels = self.transformer(
                src_input_id=batch['src_input_ids'],
                tgt_input_id=batch['tgt_input_ids'],
                pred_counts=False,
            )
            count_output = self.transformer(
                src_input_id=batch['src_input_ids'],
                tgt_input_id=batch['tgt_input_ids'],
                pred_counts=True,
            )
            return logits, labels, count_output
        elif self.return_cls_embedding:
            embeddings = self.transformer(
                src_input_id=batch['src_input_ids'],
                tgt_input_id=batch['tgt_input_ids'],
                return_cls_embedding=True,
            )
            return embeddings
        elif self.generate:
            outputs = self.transformer.generate(
                src_input_id=batch['src_input_ids'],
                tgt_input_id=batch['tgt_input_ids'],
                seq_length=batch['tgt_input_ids'].shape[1],
                tgt_vocab_size=self.tgt_vocab_size,
                noise_schedule=cosine_schedule,
            )
            return outputs

    def configure_optimizers(self):
        parameters = [{'params': self.transformer.parameters(), 'lr': self.lr}]
        optimizer = optim.Adam(parameters, weight_decay=self.weight_decay)
        # lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer,
        #     mode='min',
        #     patience=self.lr_scheduler_patience,
        # )
        return {
            'optimizer': optimizer,
            # "lr_scheduler": lr_scheduler,
            # "monitor": "train/loss",
        }

    def training_step(self, batch, *args, **kwargs):
        logits, labels, count_output = self.forward(batch)
        count_loss = self.compute_count_loss(count_output, batch)

        perp = Perplexity(ignore_index=-100).to('cuda')  # -100 = masked labels
        perp.update(logits, labels)
        logits = logits.contiguous().view(-1, logits.size(-1))
        labels = labels.contiguous().view(-1)

        masking_loss = self.masking_loss(logits, labels)

        self.log(
            'train/masking_loss',
            masking_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            'train/count_loss',
            count_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        # combined_loss = self.alpha * masking_loss + (1 - self.alpha) * count_loss
        # rescale with hyperparameter so that they have similar magnitude

        self.log(
            'train/Perplexity',
            perp.compute(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return count_loss

    def on_train_epoch_end(self):
        # return F1 score and accuracy
        pass

    def test_step(self, batch, *args, **kwargs):
        if self.return_cls_embedding:
            embeddings = self.forward(batch)
            self.cls_embeddings_list.append(embeddings[:, 0, :])
            self.tgt_output['cell_type'].append(batch['tgt_cell_type'])
            self.tgt_output['cell_population'].append(batch['tgt_cell_population'])
            self.time_point = batch['tgt_time_point']
        if self.generate:
            # num_samples = 10
            # x = torch.tensor([0])  # start with padding token
            # x.to('cuda')
            # x = x.expand(num_samples, -1)
            batch_size, sequence_length = batch['tgt_input_ids'].shape
            output, true_output, logits = self.transformer.generate(
                src_input_id=batch['src_input_ids'],
                tgt_input_id=batch['tgt_input_ids'],
                seq_length=sequence_length,
                tgt_vocab_size=self.tgt_vocab_size,
                noise_schedule=cosine_schedule
                # top_k=5
            )
            ranked_tokenised_output = batch_token_ranking(
                output, self.tgt_vocab_size
            ).float()
            ranked_tokenised_true_output = batch_token_ranking(
                true_output, self.tgt_vocab_size
            ).float()
            # ignore padding
            ranked_tokenised_output = ranked_tokenised_output.T[1:, :]
            ranked_tokenised_true_output = ranked_tokenised_true_output.T[1:, :]
            # ignore ranks when it is sequence
            mask = (ranked_tokenised_output != sequence_length) | (
                ranked_tokenised_true_output != sequence_length
            )
            spearman = self.metric['spearman'](
                ranked_tokenised_output, ranked_tokenised_true_output
            )
            spearman_mean = torch.mean(spearman)
            print(spearman_mean)
            mse = self.metric['mse'](
                ranked_tokenised_output[mask], ranked_tokenised_true_output[mask]
            )
            print(mse)
            self.log(
                'test/spearman',
                spearman_mean,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            self.log(
                'test/mse', mse, on_step=True, on_epoch=True, prog_bar=True, logger=True
            )

    def on_test_epoch_end(self):
        # return F1 score and accuracy
        # print(self.cls_embeddings)
        if self.return_cls_embedding:
            self.cls_embeddings_list = torch.cat(self.cls_embeddings_list)
            self.cls_embeddings_list = self.cls_embeddings_list.detach().cpu().numpy()
            for key in self.tgt_output:
                print(key)
                self.tgt_output[key] = np.concatenate(self.tgt_output[key], axis=0)
            cell_names = [
                'cell' + str(i) for i in range(len(self.tgt_output['cell_type']))
            ]
            obs = pd.DataFrame(
                {
                    'cell_names': cell_names,
                    'cell_type': self.tgt_output['cell_type'],
                    'cell_population': self.tgt_output['cell_population'],
                }
            )

            time_points = np.unique(self.time_point)[0]
            # adata = sc.read_h5ad(
            #     '/lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/'
            #     'T_perturb/T_perturb/pp/res/h5ad_data/cytoimmgen_tokenisation_degs_random_16h.h5ad')
            # adata.obsm['X_CLS_embeddings'] = self.cls_embeddings
            adata = ad.AnnData(X=self.cls_embeddings_list, obs=obs)
            # save anndata
            adata.write_h5ad(
                f'/lustre/scratch123/hgi/projects/healthy_imm_expr/'
                f't_generative/T_perturb/T_perturb/'
                f'plt/res/scConformer/cls_embeddings_{time_points}.h5ad'
            )
