import pickle
from typing import Dict, List

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
from torchaudio.functional import edit_distance
from torchmetrics.classification import Accuracy, F1Score

from T_perturb.Modules.T_model import TTransformer, cosine_schedule

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
        generate=True,
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
                'accuracy': Accuracy(num_classes=tgt_vocab_size, task='multiclass'),
                'f1': F1Score(num_classes=tgt_vocab_size, task='multiclass'),
            }
        )

        with open(TOKEN_DICTIONARY_FILE, 'rb') as f:
            gene_token_dict = pickle.load(f)
        self.gene_token_dict = {value: key for key, value in gene_token_dict.items()}
        # return embedding + metadata during inference
        self.return_cls_embedding = return_cls_embedding
        self.generate = generate
        self.cls_embeddings_list: List[torch.tensor] = []
        self.tgt_output: Dict[str, List[str]] = {'cell_type': [], 'cell_population': []}
        self.time_point = None

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
            output, true_output = self.transformer.generate(
                src_input_id=batch['src_input_ids'],
                tgt_input_id=batch['tgt_input_ids'],
                seq_length=246,
                tgt_vocab_size=704,
                noise_schedule=cosine_schedule
                # top_k=5
            )
            padding_token_id = 0
            print('output', output[:20, :20])
            print('true_output', true_output[:20, :20])
            mask = true_output != padding_token_id
            masked_output = output[mask]
            masked_true_output = true_output[mask]
            levenshtein_distance = edit_distance(masked_output, masked_true_output)
            print('levenshtein_distance', levenshtein_distance)
            # print('output', output[:20, :20])
            # print(output.shape==true_output.shape)
            # print('true_output', true_output[:20, :20])
            # compute Spearman correlation

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
