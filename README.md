 [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
 ![python](https://img.shields.io/badge/Python-3.11-brightgreen)


# PerturbGen Foundation model for dynamic cellular states

![PerturbGen overview](docs/fig1.png)
PerturbGen is a trajectory-aware, generative transformer for predicting how genetic or signaling perturbations reshape transcriptional programs across cell state transitions, rather than only within a single fixed state. It formulates time, development, or lineage progression as a sequence-to-sequence (S2S) problem: given an unperturbed source cell state (and optional context from earlier observed/predicted conditions), the model predicts a target downstream state (e.g., a later time point or more differentiated fate), enabling perturbation effects introduced early or mid-trajectory to be propagated to terminal outcomes. Architecturally, PerturbGen uses an encoder–decoder Transformer: a pretrained 12-layer encoder embeds the source gene-token sequence, while a Transformer decoder (dataset-tuned depth) combines self-attention over the target sequence with cross-attention to the encoded source and contextual condition embeddings. Cells are tokenized as ranked gene sequences using a Geneformer-style tokenizer (gene ranks from normalized expression), and decoding is non-autoregressive, following a MaskGIT-style iterative refinement scheme that fills masked tokens in parallel with bidirectional context, which better matches the non-sequential nature of gene regulation than left-to-right generation. Training combines (i) a masked token prediction objective on the target condition conditioned on source + context, and (ii) an auxiliary count reconstruction head (a lightweight MLP with a probabilistic count model) to map learned embeddings back to gene expression space for interpretable transcript-level perturbation readouts. At inference, PerturbGen runs paired forward passes (control vs perturbed) where perturbations are simulated by editing the source gene-token sequence (e.g., KO by removing/zeroing a token; activation by rank promotion), producing predicted downstream expression changes and embedding shifts that can be scaled to genome-wide in silico perturbation atlases across trajectories.

## 1. Usage

First, clone the repo and change to the project directory.

```shell
git clone https://github.com/Lotfollahi-lab/Perturbgen.git
```

Install Poetry (one-time):
(wanna know what is poetry? have a look at https://python-poetry.org)
```shell
curl -sSL https://install.python-poetry.org | python3 -
```
Optional: alternative way to install poetry using pipx (https://pipx.pypa.io/stable/installation/)
```shell
pipx install poetry
```

Create/install the environment and dependencies:
```shell
cd Perturbgen
poetry env use python3.11
poetry install
```

Activate the enviroment
```shell
source "$(poetry env info -p)/bin/activate"
```

The project contains some jupyter notebooks, which were converted to python files
due to better handling in the repository.
These files end with `_nb.py` and can be converted back to a `.ipynb` file with
`jupytext`:

```shell
jupytext --to ipynb --execute <your_file>_nb.py
```
## Examples

For usage, see the [documentation](https://perturbgen.cog.sanger.ac.uk/docs/examples/01_preprocessing_curation.html) or the local notebooks:
- [Preprocessing and data curation](docs/examples/01_preprocessing_curation.ipynb)
- [Tokenization and pairing](docs/examples/02_tokenization_pairing.ipynb)
- [Train PerturbGen](docs/examples/03_train_perturbgen.ipynb)
- [Gene Embedding Extraction](docs/examples/04_GeneEmbedding_Extraction.ipynb)
- [Gene Program Discovery](docs/examples/05_GeneProgram_Discovery.ipynb)
- [Perturbation](docs/examples/06_perturbation.ipynb)
- [Post Perturbation Analyses](docs/examples/07_PostPerturbation_Analyses.ipynb)

See Perturbation notebook for more explaination about how to perturb a gene or list of genes

## Documentation

Full documentation and tutorials are available at: [perturbgen.cog.sanger.ac.uk](https://perturbgen.cog.sanger.ac.uk/docs/examples/01_preprocessing_curation.html)

## Citation

If you use our repository or code in your research, please cite our paper:

```
@article{chi2026predicting,
  title={Predicting how perturbations reshape cellular trajectories with PerturbGen},
  author={Chi Hao Ly, Kevin and Miraki Feriz, Adib and Isobe, Tomoya and Vahidi, Amirhossein and Vaghari, Delshad and Rostron, Anthony and Quiroga Londono, Mariana and Mende, Nicole and Vijayabaskar, MS and Moullet, Marie and others},
  journal={bioRxiv},
  pages={2026--03},
  year={2026},
  publisher={Cold Spring Harbor Laboratory}
}
```

**Paper:** [Predicting how perturbations reshape cellular trajectories with PerturbGen (bioRxiv 2026)](https://www.biorxiv.org/content/10.64898/2026.03.04.709254v1)
