from typing import List, Literal

import numpy as np

from T_perturb.Dataloaders.datamodule import CellGenDataModule, CellGenDataset


class PerturberDataModule(CellGenDataModule):
    def __init__(
        self,
        filter_mode: Literal['include', 'exclude'] = 'include',
        condition_to_perturb: List[str] | None = None,
        condition_obs_key: str | None = None,
        condition_tps: int | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.condition_obs_key = condition_obs_key
        self.condition_to_perturb = condition_to_perturb
        self.condition_tps = condition_tps
        print(
            f'Start perturbation ...\n'
            f'- Condition to perturb: {self.condition_to_perturb}\n'
            f'- Condition observation key: {self.condition_obs_key}\n'
            f'- Filter mode: {filter_mode}\n'
        )
        self.filter_mode = filter_mode

    # create a function filter huggingface dataset
    def filter_dataset(self, dataset, condition_to_perturb, filter_mode, num_proc=1):
        # create additional column to track index of cell pairing
        # index should be row index of the dataset
        dataset = dataset.map(lambda example, idx: {'index': idx}, with_indices=True)

        if filter_mode == 'include':
            dataset_filtered = dataset.filter(
                lambda example: example[self.condition_obs_key] in condition_to_perturb,
                num_proc=num_proc,
            )

        elif filter_mode == 'exclude':
            # Exclude samples where condition_obs_key is in the list
            dataset_filtered = dataset.filter(
                lambda example: example[self.condition_obs_key]
                not in condition_to_perturb,
                num_proc=num_proc,
            )
        else:
            raise ValueError(
                f"Invalid mode: {filter_mode}. Must be 'include' or 'exclude'."
            )
        if len(dataset_filtered) == 0:
            raise ValueError(
                f'No samples found with condition_obs_key '
                f'in the list: {condition_to_perturb}\n'
                f'-> Select a different conditions to perturb, e.g.\n'
                f'{np.unique(dataset[self.condition_obs_key])}\n'
                f'-> Select the right condition_tps.'
            )

        return dataset_filtered, dataset_filtered['index']

    def setup(self, stage=None):
        # filter the dataset
        if self.condition_to_perturb is not None:
            tgt_dataset_filtered, filter_idx = self.filter_dataset(
                self.tgt_datasets[f'tgt_dataset_t{self.condition_tps}'],
                self.condition_to_perturb,
                self.filter_mode,
                num_proc=self.dataloader_kwargs['num_workers'],
            )
            self.tgt_datasets[
                f'tgt_dataset_t{self.condition_tps}'
            ] = tgt_dataset_filtered
            src_dataset_filtered = self.src_dataset.select(filter_idx)
            # exclude condition_tps from all_modelling_tps
            filtering_tps = [
                tp for tp in self.all_modelling_tps if tp != self.condition_tps
            ]
            for t in filtering_tps:
                tgt_dataset_filtered = self.tgt_datasets[f'tgt_dataset_t{t}'].select(
                    filter_idx
                )
                self.tgt_datasets[f'tgt_dataset_t{t}'] = tgt_dataset_filtered
            tgt_dataset_filtered = self.tgt_datasets
        else:
            src_dataset_filtered = self.src_dataset
            tgt_dataset_filtered = self.tgt_datasets
        dataset_args = {
            'src_dataset': src_dataset_filtered,
            'tgt_datasets': tgt_dataset_filtered,
            'src_counts': self.src_counts,
            'tgt_counts_dict': self.tgt_counts_dict,
        }

        if stage == 'fit' or stage is None:
            self.all_modelling_tps = self.pred_tps
            dataset_args['time_steps'] = self.pred_tps
            if self.condition_encodings is not None:
                dataset_args['split_indices'] = self.train_indices
                dataset_args['conditions'] = (
                    self.conditions if self.condition_keys is not None else None
                )
                dataset_args['conditions_combined'] = (
                    self.conditions_combined
                    if self.condition_keys is not None
                    else None
                )
                self.train_dataset = CellGenDataset(**dataset_args)
                if self.val_indices is not None:
                    dataset_args['split_indices'] = self.val_indices
                    self.val_dataset = CellGenDataset(**dataset_args)
                else:
                    self.val_dataset = None
            else:
                dataset_args['split_indices'] = self.train_indices
                self.train_dataset = CellGenDataset(**dataset_args)
                if self.val_indices is not None:
                    dataset_args['split_indices'] = self.val_indices
                    self.val_dataset = CellGenDataset(**dataset_args)
                else:
                    self.val_dataset = None

        if stage == 'test' or stage is None:
            # use all time steps to provide as context
            self.all_modelling_tps = self.total_tps
            dataset_args['time_steps'] = self.all_modelling_tps
            dataset_args['split_indices'] = self.test_indices
            if self.condition_encodings is not None:
                dataset_args['conditions'] = (
                    self.conditions if self.condition_keys is not None else None
                )
                dataset_args['conditions_combined'] = (
                    self.conditions_combined
                    if self.condition_keys is not None
                    else None
                )
                self.test_dataset = CellGenDataset(**dataset_args)
            else:
                self.test_dataset = CellGenDataset(**dataset_args)
