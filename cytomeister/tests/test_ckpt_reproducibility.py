import os
import unittest

import torch
import torch.nn.functional as F

if os.getcwd().split('/')[-1] != 'healthy_imm_expr':
    # set working directory to root of repository
    os.chdir('/lustre/scratch126/cellgen/team361/kl11/t_generative/')


def load_checkpoint(checkpoint_path):
    # Load checkpoint using PyTorch Lightning
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    return checkpoint['state_dict']


def checkpoints_are_equal(checkpoint_path1, checkpoint_path2):
    state_dict1 = load_checkpoint(checkpoint_path1)
    state_dict2 = load_checkpoint(checkpoint_path2)

    similarity = 0.0
    total_layers = 0
    # Compare state_dicts
    for key in state_dict1.keys():
        similarity += F.cosine_similarity(
            state_dict1[key].flatten().float(),
            state_dict2[key].flatten().float(),
            dim=0,
        ).item()
        if key not in state_dict2:
            print(f'Key {key} not found in checkpoint 2.')
            return False
        if not torch.allclose(state_dict1[key], state_dict2[key], atol=1e-4):
            print(f'Difference found in key: {key}')
            return False
        total_layers += 1

    torch.set_printoptions(precision=10)
    print(f'Cosine similarity: {similarity / total_layers}')
    return True


class TestCheckpointEquality(unittest.TestCase):
    def test_masking_checkpoints_identical(self):
        # Define paths to your checkpoints compare always to baseline
        checkpoint_path1 = (
            'T_perturb/T_perturb/tests/checkpoints/'
            'test_masking_checkpoint-epoch=00.ckpt'
        )
        checkpoint_path2 = (
            'T_perturb/T_perturb/tests/checkpoints/'
            'test_masking_checkpoint-epoch=00-v1.ckpt'
        )
        # Check if the checkpoints exist
        self.assertTrue(
            os.path.exists(checkpoint_path1),
            f'Checkpoint {checkpoint_path1} does not exist.',
        )
        self.assertTrue(
            os.path.exists(checkpoint_path2),
            f'Checkpoint {checkpoint_path2} does not exist.',
        )
        # Assert that the checkpoints are identical
        self.assertTrue(
            checkpoints_are_equal(checkpoint_path1, checkpoint_path2),
            'Masking checkpoints are not identical.',
        )

    def test_counts_checkpoints_identical(self):
        # Define paths to your checkpoints
        checkpoint_path1 = (
            'T_perturb/T_perturb/tests/checkpoints/test_counts_checkpoint-epoch=00.ckpt'
        )
        checkpoint_path2 = (
            'T_perturb/T_perturb/tests/checkpoints/'
            'test_counts_checkpoint-epoch=00-v1.ckpt'
        )
        # Check if the checkpoints exist
        self.assertTrue(
            os.path.exists(checkpoint_path1),
            f'Checkpoint {checkpoint_path1} does not exist.',
        )
        self.assertTrue(
            os.path.exists(checkpoint_path2),
            f'Checkpoint {checkpoint_path2} does not exist.',
        )
        # Assert that the checkpoints are identical
        self.assertTrue(
            checkpoints_are_equal(checkpoint_path1, checkpoint_path2),
            'Count checkpoints are not identical.',
        )


if __name__ == '__main__':
    unittest.main()
