About
=====

Model Description
-----------------

**Geneformer** is a context-aware, attention-based deep learning model pretrained on a large-scale corpus of ~30 million single-cell transcriptomes to enable context-specific predictions in settings with limited data in network biology. During pretraining, Geneformer gained a fundamental understanding of network dynamics, encoding network hierarchy in the attention weights of the model in a completely self-supervised manner. With both zero-shot learning and fine-tuning with limited task-specific data, Geneformer consistently boosted predictive accuracy in a diverse panel of downstream tasks relevant to chromatin and network dynamics. In silico perturbation with zero-shot learning identified a novel transcription factor in cardiomyocytes that we experimentally validated to be critical to their ability to generate contractile force. In silico treatment with limited patient data revealed candidate therapeutic targets for cardiomyopathy that we experimentally validated to significantly improve the ability of cardiomyocytes to generate contractile force in an iPSC model of the disease. Overall, Geneformer represents a foundational deep learning model pretrained on ~30 million human single cell transcriptomes to gain a fundamental understanding of gene network dynamics that can now be democratized to a vast array of downstream tasks to accelerate discovery of key network regulators and candidate therapeutic targets.

In `our manuscript <https://rdcu.be/ddrx0>`_, we report results for the 6 layer Geneformer model pretrained on Genecorpus-30M. We additionally provide within the repository a 12 layer Geneformer model, scaled up with retained width:depth aspect ratio, also pretrained on Genecorpus-30M.

Both the `6 <https://huggingface.co/ctheodoris/Geneformer/blob/main/pytorch_model.bin>`_ and `12 <https://huggingface.co/ctheodoris/Geneformer/blob/main/geneformer-12L-30M/pytorch_model.bin>`_ layer Geneformer models were pretrained in June 2021.

Application
-----------

The pretrained Geneformer model can be used directly for zero-shot learning, for example for in silico perturbation analysis, or by fine-tuning towards the relevant downstream task, such as gene or cell state classification.

Example applications demonstrated in `our manuscript <https://rdcu.be/ddrx0>`_ include:

| *Fine-tuning*:
| - transcription factor dosage sensitivity
| - chromatin dynamics (bivalently marked promoters)
| - transcription factor regulatory range
| - gene network centrality
| - transcription factor targets
| - cell type annotation
| - batch integration
| - cell state classification across differentiation
| - disease classification
| - in silico perturbation to determine disease-driving genes
| - in silico treatment to determine candidate therapeutic targets

| *Zero-shot learning*:
| - batch integration
| - gene context specificity
| - in silico reprogramming
| - in silico differentiation
| - in silico perturbation to determine impact on cell state
| - in silico perturbation to determine transcription factor targets
| - in silico perturbation to determine transcription factor cooperativity

Citation
--------

| C V Theodoris #, L Xiao, A Chopra, M D Chaffin, Z R Al Sayed, M C Hill, H Mantineo, E Brydon, Z Zeng, X S Liu, P T Ellinor #. `Transfer learning enables predictions in network biology. <https://rdcu.be/ddrx0>`_ *Nature*, 31 May 2023. (# co-corresponding authors)
