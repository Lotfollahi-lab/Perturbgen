"""Tests for the config that ``PerturbGen.perturb()`` builds (no GPU or data needed)."""
import os
import sys
import tempfile
import unittest
from unittest import mock

import yaml

from perturbgen.model import PerturbGen


def _make_tokenized_dir(root, source='src', n_hvg=10):
    # empty files with the layout PerturbGen() checks for
    sfx = f'{n_hvg}_hvg'
    for d in (f'dataset_{sfx}_tgt', f'h5ad_pairing_{sfx}_tgt'):
        os.makedirs(os.path.join(root, d))
    for f in (
        f'dataset_{sfx}_src/{source}.dataset',
        f'h5ad_pairing_{sfx}_src/{source}.h5ad',
        f'token_id_to_genename_{sfx}.pkl',
        f'tokenid_to_rowid_{sfx}.pkl',
    ):
        os.makedirs(os.path.dirname(os.path.join(root, f)), exist_ok=True)
        open(os.path.join(root, f), 'w').close()


class TestPerturbConfig(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        tok = os.path.join(self.tmp.name, 'tok')
        _make_tokenized_dir(tok)
        encoder = os.path.join(self.tmp.name, 'encoder.ckpt')
        open(encoder, 'w').close()
        self.model = PerturbGen(
            tok, source='src', n_hvg=10, encoder_path=encoder,
            pred_tps=[1, 2], var_list=['cell_type', 'cell_pairing_index'],
        )
        self.ckpt = os.path.join(self.tmp.name, 'count.ckpt')

    def _run(self, **kwargs):
        # replace Perturb/val.py's main with one that records the config it is given
        captured = {}

        def fake_main():
            with open(sys.argv[sys.argv.index('--config') + 1]) as f:
                captured.update(yaml.safe_load(f))

        # importing Perturb/val.py changes the working directory; restore it for other tests
        cwd = os.getcwd()
        try:
            with mock.patch('perturbgen.Perturb.val.main', fake_main):
                kwargs.setdefault('genes', ['ENSG00000125538'])
                self.model.perturb(
                    count_ckpt=self.ckpt,
                    output_dir=os.path.join(self.tmp.name, 'out'), **kwargs,
                )
        finally:
            os.chdir(cwd)
        return captured

    def test_extra_config_is_merged(self):
        config = self._run(extra_config={
            'data': {'filter_var': 'drug', 'filter_cond': ['Drug10']},
            'trainer': {'exclude_src': True},
            'datamodule': {'num_workers': 0},
        })
        self.assertEqual(config['data']['filter_var'], 'drug')
        self.assertEqual(config['data']['filter_cond'], ['Drug10'])
        self.assertTrue(config['trainer']['exclude_src'])
        self.assertEqual(config['datamodule']['num_workers'], 0)
        # generated keys are kept
        self.assertEqual(config['trainer']['genes_to_perturb'], ['ENSG00000125538'])
        self.assertEqual(config['data']['src_dataset_file'], self.model.paths['src_dataset'])

    def test_without_extra_config(self):
        config = self._run()
        self.assertNotIn('filter_var', config['data'])
        self.assertEqual(config['trainer']['validation_mode'], 'inference')
        self.assertEqual(config['datamodule']['num_workers'], 4)

    def test_relative_genes_file_is_made_absolute(self):
        config = self._run(genes=[], extra_config={'data': {
            'perturb_genes_file': 'genes.csv', 'perturb_colname': 'leiden', 'perturb_cluster': [3],
        }})
        self.assertEqual(config['data']['perturb_genes_file'], os.path.abspath('genes.csv'))

    def test_keys_set_by_perturb_are_rejected(self):
        for section, key in [('trainer', 'n_samples'), ('trainer', 'output_dir'),
                             ('datamodule', 'pred_tps'), ('trainer', 'max_seq_length'),
                             ('trainer', 'num_heads')]:
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'cannot set'):
                self._run(extra_config={section: {key: 1}})

    def test_malformed_extra_config_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'extra_config sections'):
            self._run(extra_config={'trainer_config': {'exclude_src': True}})
        with self.assertRaises(TypeError):
            self._run(extra_config={'data': None})


if __name__ == '__main__':
    unittest.main()
