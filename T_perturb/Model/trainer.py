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
import wandb
from geneformer.tokenizer import TOKEN_DICTIONARY_FILE
from pytorch_lightning import LightningModule
from torchmetrics import (
    CosineSimilarity,
    MeanSquaredError,
    PearsonCorrCoef,
    SpearmanCorrCoef,
)
from torchmetrics.text import Perplexity

from T_perturb.Modules.T_model import (
    CountDecoder,
    cosine_schedule,
    scConformer,
)
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


class scConformertrainer(LightningModule):
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
        return_embeddings: bool = False,
        generate: bool = True,
        batch_size: int = 32,
        dataset_info: Optional[str] = None,
        adata: Optional[ad.AnnData] = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.transformer = scConformer(
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
        self.lr_scheduler_factor = lr_scheduler_factor
        self.metric = nn.ModuleDict(
            {
                'perplexity': Perplexity(ignore_index=-100),
                'cosine_similarity': CosineSimilarity(reduction='mean'),
                'mse': MeanSquaredError(),
                'spearman': SpearmanCorrCoef(num_outputs=batch_size),
            }
        )

        with open(TOKEN_DICTIONARY_FILE, 'rb') as f:
            gene_token_dict = pickle.load(f)
        self.gene_token_dict = {value: key for key, value in gene_token_dict.items()}
        with open(
            '/lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/T_perturb/'
            'T_perturb/pp/res/dataset/token_dictionary_for_subset_token_id.pkl',
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
        if self.training:
            outputs = self.transformer(
                src_input_id=batch['src_input_ids'],
                tgt_input_id=batch['tgt_input_ids'],
                original_lens=batch['src_length'],
            )
            return outputs
        elif self.return_embeddings:
            outputs = self.transformer(
                src_input_id=batch['src_input_ids'],
                tgt_input_id=batch['tgt_input_ids'],
                return_embeddings=True,
            )
            return outputs
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
            'train/masking_loss',
            masking_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch['tgt_input_ids'].shape[0],
        )

        self.log(
            'train/Perplexity',
            perp.compute(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch['tgt_input_ids'].shape[0],
        )
        return masking_loss

    def on_train_epoch_end(self):
        # return F1 score and accuracy
        pass

    def test_step(self, batch, *args, **kwargs):
        if self.return_embeddings:
            outputs = self.forward(batch)
            self.cls_embeddings_list.append(outputs['embeddings'][:, 0, :])
            self.gene_embeddings_list.append(outputs['embeddings'][:, 1:, :])
            self.token_id_list.append(batch['tgt_input_ids'])
            # self.tgt_output['cell_type'].append(batch['tgt_cell_type'])
            # self.tgt_output['cell_population'].append(batch['tgt_cell_population'])
            # self.time_point = batch['tgt_time_point']
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
        if self.return_embeddings:
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
                print(emb.shape)
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
                f'plt/res/scConformer/'
                f'cls_embeddings_{self.dataset_info}_cosine_similarity.h5ad'
            )


class CountDecodertrainer(LightningModule):
    def __init__(
        self,
        ckpt_path: str = 'ckpt_path',
        loss_mode: str = 'mse',
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        lr_scheduler_patience: float = 1.0,
        lr_scheduler_factor: float = 0.8,
        conditions: Optional[Dict[Any, Any]] = None,
        conditions_combined: Optional[List[Any]] = None,
        tgt_vocab_size: int = 25000,
        d_model: int = 256,
        cell_number: int = 143360,
        generate: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.target_device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        # Create an instance of your model
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        for keys in checkpoint['state_dict']:
            print(keys)

        pretrained_model = scConformer(
            tgt_vocab_size=checkpoint['hyper_parameters']['tgt_vocab_size'],
            d_model=checkpoint['hyper_parameters']['d_model'],
            d_ff=checkpoint['hyper_parameters']['d_ff'],
            max_seq_length=checkpoint['hyper_parameters']['max_seq_length'],
        )
        state_dict = checkpoint['state_dict']

        new_state_dict = {}
        for k, v in state_dict.items():
            k = k.replace('transformer.', '')
            new_state_dict[k] = v
        state_dict = new_state_dict
        # model = model.cuda()
        # classifier = classifier.cuda()
        # criterion = criterion.cuda()

        pretrained_model.load_state_dict(new_state_dict, strict=False)

        self.decoder = CountDecoder(
            pretrained_model=pretrained_model,
            loss_mode=loss_mode,
            tgt_vocab_size=tgt_vocab_size,
            d_model=d_model,
        )
        self.save_hyperparameters()
        self.weight_decay = weight_decay
        self.lr = lr
        self.lr_scheduler_patience = lr_scheduler_patience
        self.lr_scheduler_factor = lr_scheduler_factor
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
                'pearson': PearsonCorrCoef(num_outputs=cell_number),
            }
        )
        self.true_counts_list: List[int] = []
        self.pred_counts_list: List[int] = []
        self.generate = generate

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
            return loss

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
            return loss

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
            return loss

        else:
            raise ValueError('Loss not supported, choose either mse or zinb')

    def training_step(self, batch, *args, **kwargs):
        outputs = self.forward(batch)
        count_loss = self.compute_count_loss(outputs, batch)
        self.true_counts_list.append(batch['tgt_counts'])
        if self.loss_mode in ['mse']:
            self.pred_counts_list.append(outputs['count_lognorm'])
        if self.loss_mode in ['nb', 'zinb']:
            self.pred_counts_list.append(outputs['count_mean'])

        return count_loss

    def on_train_epoch_end(self):
        # return Pearson correlation coefficient
        true_counts = torch.cat(self.true_counts_list)
        pred_counts = torch.cat(self.pred_counts_list)
        pearson = self.metric['pearson'](pred_counts.T, true_counts.T)
        mean_pearson = torch.mean(pearson)
        self.log(
            'train/pearson',
            mean_pearson,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        mse = self.metric['mse'](pred_counts, true_counts)
        mean_mse = torch.mean(mse)
        self.log(
            'train/mse',
            mean_mse,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        # empty lists
        self.true_counts_list = []
        self.pred_counts_list = []

    def test_step(self, batch, *args, **kwargs):
        if self.generate:
            # num_samples = 10
            # x = torch.tensor([0])  # start with padding token
            # x.to('cuda')
            # x = x.expand(num_samples, -1)
            batch_size, sequence_length = batch['tgt_input_ids'].shape
            output, true_output, logits = self.decoder.generate(
                src_input_id=batch['src_input_ids'],
                tgt_input_id=batch['tgt_input_ids'],
                seq_length=sequence_length,
                tgt_vocab_size=self.tgt_vocab_size,
                noise_schedule=cosine_schedule
                # top_k=5
            )
            # pass output to
            # counts = self.decoder.decoder(output)
            # pass on test epoch end
            # run distribution metrics

    def configure_optimizers(self):
        parameters = [{'params': self.decoder.parameters(), 'lr': self.lr}]
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
