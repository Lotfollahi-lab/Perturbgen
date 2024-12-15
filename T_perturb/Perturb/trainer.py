from typing import List

import scanpy as sc
import torch

# from geneformer.tokenizer import TOKEN_DICTIONARY_FILE
from torch.nn.functional import cosine_similarity

from T_perturb.Model.trainer import CountDecoderTrainer
from T_perturb.src.utils import (  # WarmupScheduler,;
    return_pert_generation_adata,
    scale_pca,
)


class PerturberGenerationTrainer(CountDecoderTrainer):
    def __init__(
        self,
        genes_to_perturb: List[int] | None = None,
        perturbation_token: int | None = 0,
        cell_type_to_perturb: str | None = None,
        perturbation_mode: List[str] | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if perturbation_mode is not None:
            self.perturbation_mode = perturbation_mode
            self.genes_to_perturb = torch.tensor(genes_to_perturb, dtype=torch.long)
            self.perturbation_token = torch.tensor(perturbation_token, dtype=torch.long)
        else:
            self.perturbation_mode = []
        self.test_dict['cls_cosine_similarity'] = []
        self.test_dict['mean_cosine_similarity'] = []

    def test_step(self, batch, *args, **kwargs):
        tgt_input_id_dict = {}
        for i in self.total_tps:
            print(i)
            tgt_input_id_ = torch.cat(
                (
                    getattr(self, f'cls_token_{str(i)}').expand(
                        batch[f'tgt_input_ids_t{i}'].shape[0], -1
                    ),
                    batch[f'tgt_input_ids_t{i}'],
                ),
                dim=1,
            )
            tgt_input_id_dict[f'tgt_input_ids_t{i}'] = tgt_input_id_
            if len(self.perturbation_mode) > 0:
                if 'tgt' in self.perturbation_mode:
                    print('perturbating tgt')
                    perturbed_tgt = batch[f'tgt_input_ids_t{i}'].clone()
                    mask = torch.isin(
                        batch[f'tgt_input_ids_t{i}'], self.genes_to_perturb
                    )
                    perturbed_tgt[mask] = self.perturbation_token
        if len(self.perturbation_mode) > 0:
            if 'src' in self.perturbation_mode:
                print('perturbating src')
                perturbed_src = batch['src_input_ids'].clone()
                mask = torch.isin(batch['src_input_ids'], self.genes_to_perturb)
                perturbed_src[mask] = self.perturbation_token
        if self.generate:
            decoder_kwargs = {
                'tgt_input_id_dict': tgt_input_id_dict,
                'mask_scheduler': self.mask_scheduler,
                'can_remask_prev_masked': False,
                'topk_filter_thres': 0.9,
                'temperature': self.temperature,
                'iterations': self.iterations,
                'sequence_length': self.sequence_length,
            }
            print('perturbation', self.perturbation_mode)
            if len(self.perturbation_mode) > 0:
                true_outputs, true_ids_dict = self.decoder.generate(
                    src_input_id=batch['src_input_ids'],
                    **decoder_kwargs,
                )
                perturbed_outputs, perturbed_ids_dict = self.decoder.generate(
                    src_input_id=perturbed_src,
                    **decoder_kwargs,
                )
                print('true', true_ids_dict)
                print('perturbed', perturbed_ids_dict)

            else:
                true_outputs, true_ids_dict = self.decoder.generate(
                    src_input_id=batch['src_input_ids'],
                    **decoder_kwargs,
                )

            for i, time_step in enumerate(true_ids_dict.keys()):
                if len(self.perturbation_mode) > 0:
                    pred_ids = perturbed_ids_dict[time_step].detach().cpu().numpy()
                    tgt_ids = true_ids_dict[time_step].detach().cpu().numpy()
                    # compute cosine similarity between perturbed and true
                    t = i + 1
                    print(perturbed_outputs[f'cls_embedding_t{t}'].shape)
                    cls_cos_sim = cosine_similarity(
                        perturbed_outputs[f'cls_embedding_t{t}'],
                        true_outputs[f'cls_embedding_t{t}'],
                    )
                    mean_agg_cos_sim = cosine_similarity(
                        perturbed_outputs[f'mean_embedding_t{t}'],
                        true_outputs[f'mean_embedding_t{t}'],
                    )
                    self.test_dict['cls_cosine_similarity'].append(cls_cos_sim)
                    self.test_dict['mean_cosine_similarity'].append(mean_agg_cos_sim)

                else:
                    pred_ids = true_ids_dict[time_step].detach().cpu().numpy()
                    tgt_ids = batch[time_step].detach().cpu().numpy()
                if self.return_rouge_score:
                    test_dict = self.compute_rouge_score(
                        pred_ids=pred_ids,
                        tgt_ids=tgt_ids,
                        rouge_len_list=self.rouge_seq_len_list,
                        max_seq_length=self.max_seq_length,
                        test_dict=self.test_dict,
                    )
                    self.test_dict = test_dict
                    # self.log(
                    #     'test/rouge1',
                    #     rouge_score['rouge1'],
                    #     on_step=False,
                    #     on_epoch=True,
                    #     prog_bar=True,
                    #     logger=True,
                    #     rank_zero_only=True,
                    #     sync_dist=True,
                    #     batch_size=batch['src_input_ids'].shape[0],
                    # )
            for time_step in self.pred_tps:
                self.test_dict['cell_idx'].append(batch[f'tgt_cell_idx_t{time_step}'])
                if len(self.var_list) > 0:
                    for var in self.var_list:
                        self.test_dict[var].append(batch[f'{var}_t{time_step}'])
                cls_embeddings = (
                    true_outputs[f'cls_embedding_t{time_step}'].detach().cpu()
                )
                self.test_dict['cls_embeddings'].append(cls_embeddings)

    def on_test_epoch_end(self):
        if self.generate:
            obs_key = self.var_list if len(self.var_list) > 0 else []
            obs_key.extend(['cell_idx'])
            pred_adata = return_pert_generation_adata(
                test_dict=self.test_dict,
                obs_key=obs_key,
                output_dir=self.output_dir,
                file_name=(
                    f'{self.date}_generate_adata_'
                    f't{self.pred_tps}_{self.encoder}_s{self.seed}_'
                    f'l{self.loss_mode}_n{self.n_samples}'
                    f'_p{self.pos_encoding_mode}_'
                    f'm{self.mask_scheduler}_s{self.sequence_length}'
                ),
            )
            # true counts are stored in the 'counts' layer
            true_adata = pred_adata.copy()
            true_adata.X = true_adata.layers['counts']
            # log norm and compute PCA
            pred_adata = scale_pca(pred_adata)
            true_adata = scale_pca(true_adata)
            # scale pca
            coords = true_adata.obsm['X_pca']
            coords = (coords - coords.mean(axis=0)) / coords.std(axis=0)
            true_adata.obsm['X_pca_scaled'] = coords
            coords = pred_adata.obsm['X_pca']
            coords = (coords - coords.mean(axis=0)) / coords.std(axis=0)
            pred_adata.obsm['X_pca_scaled'] = coords

            # subsample 25k cells
            if pred_adata.shape[0] > 10000:
                sc.pp.subsample(pred_adata, n_obs=10000, copy=False)
                # use obs index to subsample true counts
                true_adata = true_adata[pred_adata.obs.index]
