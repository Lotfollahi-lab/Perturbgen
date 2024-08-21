Getting Started
===============

Installation
------------

Geneformer installation instructions.

Make sure you have git-lfs installed (https://git-lfs.com).

.. code-block:: bash

    git lfs install
    git clone https://huggingface.co/ctheodoris/Geneformer
    cd Geneformer
    pip install .


Tutorials
---------

| See `examples <https://huggingface.co/ctheodoris/Geneformer/tree/main/examples>`_ for:
| - tokenizing transcriptomes
| - pretraining
| - hyperparameter tuning
| - fine-tuning
| - extracting and plotting cell embeddings
| - in silico perturbation

Please note that the fine-tuning examples are meant to be generally applicable and the input datasets and labels will vary dependent on the downstream task. Example input files for a few of the downstream tasks demonstrated in the manuscript are located within the `example_input_files directory <https://huggingface.co/datasets/ctheodoris/Genecorpus-30M/tree/main/example_input_files>`_ in the dataset repository, but these only represent a few example fine-tuning applications.


Tips
----

Please note that GPU resources are required for efficient usage of Geneformer. Additionally, we strongly recommend tuning hyperparameters for each downstream fine-tuning application as this can significantly boost predictive potential in the downstream task (e.g. max learning rate, learning schedule, number of layers to freeze, etc.).
