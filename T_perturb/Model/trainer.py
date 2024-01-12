import pickle

import scanpy as sc

# import torch
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

    def forward(self, batch):
        src_batch = batch['src']
        tgt_batch = batch['tgt']
        output, labels = self.transformer(
            src_input_id=src_batch['input_id'],
            src_attention_mask=src_batch['attention_mask'],
            tgt_input_id=tgt_batch['input_id'],
        )
        return output, labels

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
        output = output.view(-1, output.size(-1))
        labels = labels.view(-1)
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
        output, labels = self.forward(batch)  # adapt based on output
        output = output.view(-1, output.size(-1))
        labels = labels.view(-1)
        return output, labels
