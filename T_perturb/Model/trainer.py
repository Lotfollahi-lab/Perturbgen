import pickle
from typing import List

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from geneformer.tokenizer import TOKEN_DICTIONARY_FILE
from pytorch_lightning import LightningModule

from T_perturb.Modules.T_model import TTransformer

wandb.init(
    entity='k-ly',
    project='ttransformer',
    dir='/lustre/scratch123/hgi/projects/healthy_imm_expr/'
    't_generative/T_perturb/T_perturb',
)
sc.settings.set_figure_params(dpi=500)


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
        return_cls_embedding=False,
        *args,
        **kwargs
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

        with open(TOKEN_DICTIONARY_FILE, 'rb') as f:
            gene_token_dict = pickle.load(f)
        self.gene_token_dict = {value: key for key, value in gene_token_dict.items()}
        self.cls_embeddings: List[torch.tensor] = []
        self.tgt_cell_type: List[str] = []
        self.tgt_cell_population: List[str] = []
        self.adata_path = None
        self.return_cls_embedding = return_cls_embedding

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
        return loss

    def on_train_epoch_end(self):
        # return F1 score and accuracy
        pass

    def test_step(self, batch, *args, **kwargs):
        # return embedding
        if self.return_cls_embedding:
            _, embeddings = self.forward(batch)
            self.cls_embeddings.append(embeddings[:, 0, :])
            self.tgt_cell_type.append(batch['tgt_cell_type'])
            self.tgt_cell_population.append(batch['tgt_cell_population'])

        # num_samples = 10
        # x = torch.tensor([0]) #start with padding token
        # x.to('cuda')
        # x = x.expand(num_samples, -1)
        # output = self.transformer.generate(
        #     src_input_id=batch['tgt_input_ids'],
        #     tgt_input_id=batch['tgt_input_ids'],
        #     seq_length=246,
        #     threshold=0.8,
        #     # tgt_vocab_size=704
        #     # top_k=5
        # )

        # print('output', output[:20, :20])

    def on_test_epoch_end(self):
        # return F1 score and accuracy
        # print(self.cls_embeddings)
        if self.return_cls_embedding:
            self.cls_embeddings = torch.cat(self.cls_embeddings)
            self.cls_embeddings = (
                self.cls_embeddings.detach().cpu().numpy()
            )  # Convert to numpy array
            self.tgt_cell_type = np.concatenate(self.tgt_cell_type)
            self.tgt_cell_population = np.concatenate(self.tgt_cell_population)
            cell_names = ['cell' + str(i) for i in range(len(self.tgt_cell_type))]
            obs = pd.DataFrame(
                {
                    'cell_names': cell_names,
                    'tgt_cell_type': self.tgt_cell_type,
                    'tgt_cell_population': self.tgt_cell_population,
                }
            )
            # adata = sc.read_h5ad(
            #     '/lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/'
            #     'T_perturb/T_perturb/pp/res/h5ad_data/cytoimmgen_tokenisation_degs_random_16h.h5ad')
            # adata.obsm['X_CLS_embeddings'] = self.cls_embeddings
            adata = ad.AnnData(X=self.cls_embeddings, obs=obs)
            # save anndata
            adata.write_h5ad(
                '/lustre/scratch123/hgi/projects/healthy_imm_expr/'
                't_generative/T_perturb/T_perturb/pp/res/cls_embeddings.h5ad'
            )
