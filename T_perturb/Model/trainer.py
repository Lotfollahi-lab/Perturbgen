import itertools

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pytorch_lightning import LightningModule
import pandas as pd
from module import Projectionhead
from utils import mean_nonpadding_embs
from geneformer.in_silico_perturber import quant_layers
import torch
import scanpy as sc
import wandb
from geneformer.tokenizer import TOKEN_DICTIONARY_FILE
import pickle
import matplotlib.pyplot as plt
wandb.init(
            entity='amirh-vahidi',
            project='clip_gene_seq',

            dir="/lustre/scratch126/cellgen/team205/av13/clipgeneseq",
        )
sc.settings.set_figure_params(dpi=500)



class Clipgenetrainer(LightningModule):
    def __init__(
        self,
        # gene
        gene_encoder,
        gene_embedding_dims: int = 256,
        seq_embedding_dims: int = 3072,
        projection_dims: int = 256,
        dropout: float = 0.0,
        temperature: float = 1.0,
        weight_decay: float = 0.0,
        head_lr: float = 1e-3,
        gene_encoder_lr: float = 1e-4,
        lr_scheduler_patience: float = 1.0,
        lr_scheduler_factor: float = 0.8,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.gene_encoder = gene_encoder
        self.gene_projection = Projectionhead(
            in_dim=gene_embedding_dims,
            out_dim=projection_dims,
        )
        self.seq_projection = Projectionhead(
            in_dim=seq_embedding_dims,
            out_dim=projection_dims,
        )
        self.temperature = temperature
        self.weight_decay = weight_decay
        self.head_lr = head_lr
        self.gene_encoder_lr = gene_encoder_lr
        self.lr_scheduler_patience = lr_scheduler_patience
        self.layer_to_quant = quant_layers(gene_encoder) + -1
        self.save_hyperparameters()
        self.embeddings = []
        self.cell_type = []
        self.length = []
        self.input_id =[]

        with open(TOKEN_DICTIONARY_FILE, "rb") as f:
                gene_token_dict = pickle.load(f)
        self.gene_token_dict = {value: key for key, value in gene_token_dict.items()}

    def _compute_losses(self, gene_embeddings, seq_embeddings):
        seq_embeddings = seq_embeddings.view(1,-1,seq_embeddings.shape[2]).squeeze()
        gene_embeddings = gene_embeddings.view(1, -1, gene_embeddings.shape[2]).squeeze()
        logits = (seq_embeddings @ gene_embeddings.T) / self.temperature
        gene_similarity = gene_embeddings @ gene_embeddings.T
        seq_similarity = seq_embeddings @ seq_embeddings.T
        targets = F.softmax(
            (gene_similarity + seq_similarity) / 2 * self.temperature, dim=-1
        )
        gene_loss = (-targets.T * nn.functional.log_softmax(logits.T,dim=-1)).sum(1)
        seq_loss = (-targets * nn.functional.log_softmax(logits,dim=-1)).sum(1)
        return (gene_loss + seq_loss) / (2.0*2048)

    def forward(self, batch):
        features = self.gene_encoder(
            input_ids=batch["input_id"].cuda(),
            attention_mask=batch["attention_mask"].cuda()
        )
        gene_features = features.hidden_states[self.layer_to_quant]

        seq_features = batch['enformer_emb']
        gene_embeddings = self.gene_projection.forward(gene_features)
        seq_embeddings = self.seq_projection.forward(seq_features)

        return gene_embeddings, seq_embeddings

    def configure_optimizers(self):
        parameters = [
            {"params": self.gene_encoder.parameters(), "lr": self.gene_encoder_lr},
            {
                "params": itertools.chain(
                    self.gene_projection.parameters(),
                    self.seq_projection.parameters(),
                ),
                "lr": self.head_lr,
                "weight_decay": self.weight_decay,
            },
        ]
        optimizer = optim.Adam(parameters, weight_decay=self.weight_decay)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=self.lr_scheduler_patience,
        )
        return {
            "optimizer": optimizer,
            # "lr_scheduler": lr_scheduler,
            # "monitor": "train/loss",
        }

    def training_step(self, batch, *args, **kwargs):
        gene_embeddings, seq_embeddings = self.forward(batch)
        loss = self._compute_losses(gene_embeddings, seq_embeddings).mean()
        train_loss = self.all_gather(loss)
        self.log("train/loss", train_loss.mean())
        return loss

    def test_step(self, batch, *args, **kwargs):
        embeddings, seq_embeddings = self.forward(batch)
        self.embeddings.append(embeddings)
        self.cell_type.append(batch["cell_type"])
        self.length.append(batch["length"])
        self.input_id.append(batch["input_id"].cpu())

    def on_test_epoch_end(self):
        cell_embedding = np.zeros((len(self.length),256))
        for i in range(len(self.length)):
            cell_embedding[i] = mean_nonpadding_embs(self.embeddings[i],self.length[i])
        self.input_id = np.array(self.input_id).flatten()
        gene_embeddings = torch.stack(self.embeddings).squeeze(1).view(1, -1, 256).squeeze()
        gene_embeddings = gene_embeddings[self.input_id > 0, :].cpu().numpy()
        self.input_id = self.input_id[self.input_id > 0]

        ensembl_id = [self.gene_token_dict.get(x) for x in self.input_id if x in self.gene_token_dict]

        andata =sc.read_h5ad('./dataset/pbmc10x_3k/preprocess_pbmc10x_3k.h5ad')
        andata.obsm["cell_embedding"] = cell_embedding
        emb_adata_test = sc.AnnData(X=cell_embedding, obs=andata.obs)
        emb_adata_test.write_h5ad('./dataset/pbmc10x_3k/cellresult_pbmc10x_3k.h5ad')


        emb_adata_test_gene = sc.AnnData(X=gene_embeddings
                                    , obs={"ensembl_id": ensembl_id}
                                    )
        emb_adata_test_gene.write_h5ad('./dataset/pbmc10x_3k/generesult_pbmc10x_3k.h5ad')

        sc.pp.neighbors(emb_adata_test_gene)
        sc.tl.umap(emb_adata_test_gene)
        fig, ax = plt.subplots(figsize=(1, 1))
        sc.pl.umap(emb_adata_test_gene, color='ensembl_id', ax=ax, palette='jet', save='umap_plot.png')
        plot_path = '/lustre/scratch126/cellgen/team205/av13/clipgeneseq/result/umap_plot.png'
        fig.savefig(plot_path, dpi=300)
        wandb.log({"plot": wandb.Image(fig)})