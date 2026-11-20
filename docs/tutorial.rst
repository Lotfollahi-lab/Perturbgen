Tutorial
========

Both tutorials use the in vivo LPS challenge data from the manuscript. They assume familiarity with Python-based
scientific computing and single-cell data.

Example 1: the full PerturbGen pipeline
---------------------------------------

**Start here.** This notebook runs the complete workflow with the Python API: tokenize the data, train the masking
and count models, extract embeddings, then knock out *IL1B* at 6 h and predict the 10 h state.

.. toctree::
   :maxdepth: 1

   examples/train_and_perturb

Example 2: perturbation options (optional)
------------------------------------------

This notebook explains the perturbation options in more detail (``genes``, ``mode``, ``where``, ``pred_tps``,
``pert_tps`` and ``extra_config``) and describes the output. It uses the released *IL1B* model from the manuscript,
which perturbs the 90 min LPS state and predicts the 6 h and 10 h states, downloaded from Hugging Face, so no
training is needed.

.. toctree::
   :maxdepth: 1

   examples/perturbation
