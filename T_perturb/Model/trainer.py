import pickle
from typing import (
    Dict,
    List,
    Union,
)

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from geneformer.tokenizer import TOKEN_DICTIONARY_FILE
from losses import (
    mse_loss,
    nb,
    zinb,
)
from pytorch_lightning import LightningModule
from torchmetrics import (
    CosineSimilarity,
    MeanSquaredError,
    SpearmanCorrCoef,
)
from torchmetrics.text import Perplexity

from T_perturb.Modules.T_model import TTransformer, cosine_schedule

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
        d_model: int = 256,  # change hyperparameter
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
        loss_mode: Union[str, str, str] = 'zinb',
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
        )
        self.loss = nn.CrossEntropyLoss()
        self.save_hyperparameters()
        self.weight_decay = weight_decay
        self.lr = lr
        self.lr_scheduler_patience = lr_scheduler_patience
        self.lr_scheduler_factor = lr_scheduler_factor
        print('tgt_vocab_size', tgt_vocab_size)
        self.metric = nn.ModuleDict(
            {
                'perplexity': Perplexity(ignore_index=-100),
                'cosine_similarity': CosineSimilarity(reduction='mean'),
                'mse': MeanSquaredError(),
                'spearman': SpearmanCorrCoef(num_outputs=64),
            }
        )
        if loss_mode == 'mse':
            self.loss = mse_loss
        elif loss_mode == 'zinb':
            self.loss = zinb
        elif loss_mode == 'nb':
            self.loss = nb
        else:
            raise ValueError('Loss not supported, choose either mse or zinb')

        with open(TOKEN_DICTIONARY_FILE, 'rb') as f:
            gene_token_dict = pickle.load(f)
        self.gene_token_dict = {value: key for key, value in gene_token_dict.items()}
        # return embedding + metadata during inference
        self.return_cls_embedding = return_cls_embedding
        self.generate = generate
        self.cls_embeddings_list: List[torch.tensor] = []
        self.tgt_output: Dict[str, List[str]] = {'cell_type': [], 'cell_population': []}
        self.time_point = None
        self.tgt_vocab_size = tgt_vocab_size

    def forward(self, batch):
        if self.training:
            output, labels = self.transformer(
                src_input_id=batch['src_input_ids'],
                tgt_input_id=batch['tgt_input_ids'],
            )

            return output, labels
        else:
            output, embeddings = self.transformer(
                src_input_id=batch['src_input_ids'],
                tgt_input_id=batch['tgt_input_ids'],
            )
            return output, embeddings

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
        output, labels = self.forward(batch)  # adapt based on output
        # softmax_output = nn.Softmax(dim=1)(output)
        # print(labels.shape)
        # print('labels', labels[:, :5])
        # print(output.shape)
        # arg_max = torch.argmax(nn.Softmax(dim=2)(output), dim=2)
        # #reashape to bxlxtoken_size
        # print('arg_max', arg_max[:, :5])
        perp = Perplexity(ignore_index=-100).to('cuda')  # -100 = masked labels
        perp.update(output, labels)
        output = output.contiguous().view(-1, output.size(-1))
        labels = labels.contiguous().view(-1)

        loss = self.loss(output, labels)
        # correct = 0
        # __, predicted = torch.max(output, 1)
        # correct = (predicted == labels).sum().item()
        # total = labels.size(0)
        self.log(
            'train/loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            'train/Perplexity',
            perp.compute(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

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
