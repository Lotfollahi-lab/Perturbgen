Tutorial
========

All tutorials use the in vivo LPS challenge data from the manuscript. They assume familiarity with Python-based
scientific computing and single-cell data.

Example 1: the full PerturbGen pipeline
---------------------------------------

**Start here.** This notebook runs the complete workflow with the Python API on a simple setting: tokenize the data,
train the masking and count models, extract embeddings, then knock out *IL1B* at 6 h and predict the 10 h state.

.. toctree::
   :maxdepth: 1

   examples/train_and_perturb

Example 2: manuscript analyses in more detail (optional)
--------------------------------------------------------

These notebooks are not needed to use PerturbGen. They show the LPS analyses from the manuscript in more detail and
use the released checkpoints from Hugging Face, so no training is needed.

- **Data preparation:** how the public and private LPS datasets were curated and merged.
- **In silico perturbation and post-perturbation analyses:** the released *IL1B* model, which perturbs the 90 min LPS
  state and predicts the 6 h and 10 h states.
- **Cell and gene embeddings and gene program discovery:** the released embedding model, trained with the unstimulated
  state as the source and 90 min, 6 h and 10 h as targets.

.. toctree::
   :maxdepth: 1

   examples/preprocessing_curation
   examples/perturbation
   examples/post_perturbation_analyses
   examples/cell_gene_embeddings
   examples/gene_program_discovery
