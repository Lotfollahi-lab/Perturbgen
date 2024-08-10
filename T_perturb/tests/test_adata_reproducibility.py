import unittest

import numpy as np
import scanpy as sc


def load_anndata(file_path):
    # Load AnnData object from file
    return sc.read_h5ad(file_path)


def anndata_are_equal(adata1, adata2, rtol=1e-5, atol=1e-8):
    # Compare the main components of two AnnData objects
    if not np.array_equal(adata1.X, adata2.X):
        print('Expression matrices differ.')
        return False

    if not adata1.obsm.keys() == adata2.obsm.keys():
        print('Obsm keys differ.')
        return False

    if not np.array_equal(adata1.obsm['cls_embeddings'], adata2.obsm['cls_embeddings']):
        print('Obsm cls_embeddings differ.')
        return False

    if not adata1.obs.equals(adata2.obs):
        print('Observation annotations differ.')
        return False

    if not adata1.var.equals(adata2.var):
        print('Variable annotations differ.')
        return False

    for key in adata1.obsm.keys():
        if not np.allclose(adata1.obsm[key], adata2.obsm[key], rtol=rtol, atol=atol):
            print(f'Obsm data for key {key} differ.')
            return False

    if adata1.layers.keys() != adata2.layers.keys():
        print('Layer keys differ.')
        return False

    for key in adata1.layers.keys():
        if not np.allclose(
            adata1.layers[key], adata2.layers[key], rtol=rtol, atol=atol
        ):
            print(f'Layer data for key {key} differ.')
            return False

    print('AnnData objects are identical.')
    return True


class TestAnnDataEquality(unittest.TestCase):
    def test_anndata_identical(self):
        # Define paths to your AnnData files
        anndata_path1 = (
            './T_perturb/T_perturb/tests/res/'
            'baseline_adata_extrapolate_Transformer_encoder_42_zinb_1.h5ad'
        )
        anndata_path2 = './T_perturb/T_perturb/tests/res/20240810_random_embs_'
        'generate_adata_extrapolate_[1, 2]__Transformer_encoder_42_zinb_1.h5ad'

        # Load AnnData objects
        adata1 = load_anndata(anndata_path1)
        adata2 = load_anndata(anndata_path2)

        # Assert that the AnnData objects are identical
        self.assertTrue(
            anndata_are_equal(adata1, adata2), 'AnnData objects are not identical.'
        )


if __name__ == '__main__':
    unittest.main()
