 [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
 ![python](https://img.shields.io/badge/Python-3.11-brightgreen)


# PerturbGen Foundation model for dynamic cellular states

![PerturbGen overview](docs/architecture.png)

A major challenge in biology is predicting how cells transition between states over time and how perturbations disrupt these transitions. Although recent approaches can predict single-cell perturbation responses in silico, they cannot predict responses across dynamic cell trajectories—for example, how early perturbations reconfigure later cell states. **PerturbGen** is a generative foundation model trained on over 100 million single-cell transcriptomes that predicts perturbation responses along cellular trajectories. It predicts how genetic perturbation at a source state shapes downstream states, alters gene programs, and modifies trajectories across time (e.g., in differentiation or disease progression).

**Applications:** This framework supports three downstream applications (Fig. 1b). First, PerturbGen predicts gene expression at specified target states, allowing inference of intermediate and future cell states. Second, learned gene embeddings can be aggregated across biological covariates, such as time, lineage, or developmental stage, to identify de novo, context-specific gene programs beyond predefined pathway annotations. Third, PerturbGen enables in silico perturbation analysis by simulating genetic interventions across cellular states. Scaling these simulations across genes yields perturbation atlases in which perturbations with similar transcriptional effects cluster together. We define these clusters as PIPs, which facilitate systematic identification of established regulators and discovery of previously unrecognized drivers of cell state transitions.

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
