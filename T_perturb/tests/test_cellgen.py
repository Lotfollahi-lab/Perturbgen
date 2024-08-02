import unittest

import numpy as np
import pytorch_lightning as pl
import torch
from datasets import Dataset

from T_perturb.Dataloaders.datamodule import PetraDataModule
from T_perturb.Modules.T_model import Petra

'''
The unittest code was adopted from the fairseq library.
https://github.com/facebookresearch/fairseq/blob/main/tests/test_transformer.py # noqa
Accessed on 2024-08-01
'''


def dummy_src_dataset(
    max_len: int = 50,
    src_vocab_size: int = 100,
    num_samples: int = 100,
):
    src_input_ids = torch.randint(0, src_vocab_size, (num_samples, max_len))
    src_input_ids[:, -10:] = 0
    src_dataset = Dataset.from_dict(
        {'input_ids': src_input_ids, 'length': [len(src_input_ids)] * num_samples}
    )
    return src_dataset


def dummy_tgt_dataset(
    max_len: int = 10,
    tgt_vocab_size: int = 100,
    num_samples: int = 100,
):
    tgt_input_ids_t1 = torch.randint(0, tgt_vocab_size, (num_samples, max_len))
    tgt_input_ids_t2 = torch.randint(0, tgt_vocab_size, (num_samples, max_len))
    # pad token
    tgt_input_ids_t1[:, -10:] = 0
    tgt_input_ids_t2[:, -10:] = 0
    tgt_dataset_t1 = Dataset.from_dict(
        {
            'input_ids': tgt_input_ids_t1,
            'length': [len(tgt_input_ids_t1)] * num_samples,
            'cell_pairing_index': np.random.choice(100, num_samples, replace=False),
        }
    )
    tgt_dataset_t2 = Dataset.from_dict(
        {
            'input_ids': tgt_input_ids_t2,
            'length': [len(tgt_input_ids_t2)] * num_samples,
            'cell_pairing_index': np.random.choice(100, num_samples, replace=False),
        }
    )
    tgt_dataset = {
        'tgt_dataset_t1': tgt_dataset_t1,
        'tgt_dataset_t2': tgt_dataset_t2,
    }
    return tgt_dataset


class PetraTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(PetraTestCase, self).__init__(*args, **kwargs)
        self.time_step = [1, 2]
        self.total_time_steps = 2
        self.max_seq_length = 50
        self.tgt_vocab_size = 100
        self.batch_size = 4
        self.d_model = 12

    def setUp(self):
        pl.seed_everything(42)
        model = Petra(
            tgt_vocab_size=self.tgt_vocab_size,
            d_model=self.d_model,
            num_heads=4,
            num_layers=1,
            d_ff=14,
            max_seq_length=self.max_seq_length + 10,  # +10 for special tokens
            dropout=0,
            mlm_probability=0.15,
            time_steps=self.time_step,
            total_time_steps=2,
            mode='Transformer_encoder',
        )
        self.model = model
        tgt_datasets = dummy_tgt_dataset(
            max_len=self.max_seq_length,
            tgt_vocab_size=self.tgt_vocab_size,
            num_samples=100,
        )
        src_dataset = dummy_src_dataset(
            max_len=self.max_seq_length,
            src_vocab_size=self.tgt_vocab_size,
            num_samples=100,
        )
        self.data_module = PetraDataModule(
            src_dataset=src_dataset,
            tgt_datasets=tgt_datasets,
            batch_size=self.batch_size,
            num_workers=1,
            time_steps=[1, 2],
            total_time_steps=2,
            train_indices=np.random.choice(100, 80, replace=False),
            max_len=self.max_seq_length,
        )
        self.data_module.setup()
        # Store cls tokens as attributes
        self.cls_tokens = []
        for i in self.time_step:
            cls_token = torch.tensor([self.tgt_vocab_size + (i - 1)], dtype=torch.long)
            self.cls_tokens.append(cls_token)

    def test_forward_pass(self):
        # Test forward pass
        X = next(iter(self.data_module.train_dataloader()))
        # Add CLS token
        tgt_input_id_dict = {}
        for i in self.time_step:
            cls_token_ = self.cls_tokens[i - 1].expand(
                X[f'tgt_input_ids_t{i}'].shape[0], -1
            )
            tgt_input_id = torch.cat((cls_token_, X[f'tgt_input_ids_t{i}']), dim=1)
            tgt_input_id_dict[f'tgt_input_id_t{i}'] = tgt_input_id

        interval = X[f'tgt_input_ids_t{i}'].shape[1] + 1  # as 0 is cls token
        num_time_steps = len(self.time_step)
        cls_positions = np.arange(0, num_time_steps * interval, interval)
        output = self.model(
            src_input_id=X['src_input_ids'],
            tgt_input_id_dict=tgt_input_id_dict,
            cls_positions=cls_positions,
            not_masked=True,
        )
        print(output['dec_embedding'].shape)
        self.assertEqual(
            output['dec_embedding'].shape,
            (
                self.batch_size,
                len(self.time_step) * self.max_seq_length + len(self.cls_tokens),
                self.d_model,
            ),
        )


if __name__ == '__main__':
    unittest.main()


# def mk_sample(tok: Sequence[int] = None, batch_size: int = 2) -> Dict[str, Any]:
#     if not tok:
#         tok = [10, 11, 12, 13, 14, 15, 2]

#     batch = torch.stack([torch.tensor(tok, dtype=torch.long)] * batch_size)
#     sample = {
#         "net_input": {
#             "tgt_tokens": batch,
#             "prev_output_tokens": batch,
#             "tgt_lengths": torch.tensor(
#                 [len(tok)] * batch_size, dtype=torch.long, device=batch.device
#             ),
#         },
#         "target": batch[:, 1:],
#     }
#     return sample


# def mk_transformer(**extra_args: Any):
#     overrides = {
#         # Use characteristics dimensions
#         "encoder_embed_dim": 12,
#         "encoder_ffn_embed_dim": 14,
#         "decoder_embed_dim": 12,
#         "decoder_ffn_embed_dim": 14,
#         # Disable dropout so we have comparable tests.
#         "dropout": 0,
#         "attention_dropout": 0,
#         "activation_dropout": 0,
#         "encoder_layerdrop": 0,
#     }
#     overrides.update(extra_args)
#     # Overrides the defaults from the parser
#     args = argparse.Namespace(**overrides)
#     transformer.tiny_architecture(args)

#     torch.manual_seed(0)
#     task = FakeTask(args)
#     return transformer.TransformerModel.build_model(args, task)


# class TransformerTestCase(unittest.TestCase):
#     def test_forward_backward(self):
#         model = mk_transformer(encoder_embed_dim=12, decoder_embed_dim=12)
#         sample = mk_sample()
#         o, _ = model.forward(**sample["net_input"])
#         loss = o.sum()
#         loss.backward()

#     def test_different_encoder_decoder_embed_dim(self):
#         model = mk_transformer(encoder_embed_dim=12, decoder_embed_dim=16)
#         sample = mk_sample()
#         o, _ = model.forward(**sample["net_input"])
#         loss = o.sum()
#         loss.backward()
