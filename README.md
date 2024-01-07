 [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
 ![python](https://img.shields.io/badge/Python-3.10-brightgreen)

<p align="center">
<img src=assets/  alt="Mo's Lab logo"/>
</p>

# Mo's Lab projects: This repo contains projects for the Mo Lab at Sanger institute

## 0. Introduction & Scope

Introducing **lotfollibrary** 

[comment]: <> (&#40;**B**ERD's **E**ducational **A**rchive for **R**esearch on)

[comment]: <> (o**M**achine Learning&#41;:)

As more and more biology and single cell researchers rely on digital tools and methods,
the need for accessible and effective training in these areas becomes increasingly
pressing. **lotfollibrary** seeks .


### Projects

Currently available:

- [Modules](lotfollibrary/Modules/)
- 

### Discussion Board

This repository is accompanied by a discussion board intended for active communication with and among the community.
Please feel free to ask your questions there, share valuable insights and give us feedback on our material.

### Disclaimer

Please note that the contents of this repository are still in the experimental early
stages and may be subject to significant changes, bugs, and limitations.
We are continuously working on improving the **lotfollibrary** repository and welcome any
feedback or contributions. Thank you for your understanding.

## 1. Usage

First, clone the repo and change to the project directory.

```shell
git clone https://github.com/amirvhd/lotfollibrary.git
```

The relevant use-cases and source codes are located in `lotfollibrary`.
Currently, we support **python >= 3.10**.
It is recommended to install the required dependencies in a separate environment, e.g.
via `conda`.
A simpler alternative is a virtual environment, which is created and activated with:

```shell
python -m venv .venv
source .venv/bin/activate
```

Dependencies are then installed via `pip`.

```shell
pip install -r requirements.txt
```

The `lotfollibrary` project is structured like a python package, which has the advantage of
being able to **install** it and thus reuse modules or functions without worrying about
absolute filepaths.
An editable version of `lotfollibrary` is also installed over `pip`:

```shell
pip install -e .
```

The project contains some jupyter notebooks, which were converted to python files
due to better handling in the repository.
These files end with `_nb.py` and can be converted back to a `.ipynb` file with
`jupytext`:

```shell
jupytext --to ipynb --execute <your_file>_nb.py
```

The `--execute` flag triggers executing every cell during conversion.
Alternatively, you can run the `_nb.py` files like every other python script.

## 2. Contributing

New ideas and improvements are always welcome. Feel free to open an issue or contribute
over a pull request.
Our repository has a few automatic checks in place that ensure a compliance with PEP8 and static
typing.
It is recommended to use `pre-commit` as a utility to adhere to the GitHub actions hooks
beforehand.
First, install the package over pip and then set a hook:
```shell
pip install pre-commit
pre-commit install
```

To ensure code serialization and keeping the memory profile low, `.ipynb` are blacklisted
in this repository.
A notebook can be saved to the repo by converting it to a serializable format via
`jupytext`, preferably `py:percent`:

```shell
jupytext --to py:percent <notebook-to-convert>.ipynb
```

The result is a python file, which can be committed and later on be converted back to `.ipynb`.
A notebook-python file from jupytext shall carry the suffix `_nb.py`.


## Citation

If you use our repository or code in your research, please cite us:

```

```
