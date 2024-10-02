 [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
 ![python](https://img.shields.io/badge/Python-3.10-brightgreen)


# TRACE: Temporal Reconstruction of Cellular Events

## 1. Usage

First, clone the repo and change to the project directory.

```shell
git clone git@github.com:Lotfollahi-lab/T_perturb.git
```

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

The `TRACE` project is structured like a python package, which has the advantage of
being able to **install** it and thus reuse modules or functions without worrying about
absolute filepaths.
An editable version of `TRACE` is also installed over `pip`:

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

The model can be run by following the four different steps:
1. Data preprocessing, tokenisation
2. Training the masking model
3. Training the count decoder model
4. Load checkpoint and generate predictions
The executable scripts for each step are located in the `T_perturb/T_perturb/batch_job_script` directory.

Below is an example of how to run the model on the T cell dataset:
```shell
bash ./T_perturb/T_perturb/batch_job_script/cytoimmgen/1_run_GF_tokenisation.sh
```

```shell
bash ./T_perturb/T_perturb/batch_job_script/cytoimmgen/2_run_train_masking_GF_frozen_interpolation.sh
```

```shell
bash ./T_perturb/T_perturb/batch_job_script/cytoimmgen/3_run_train_count_GF_frozen_interpolation.sh
```

```shell
bash ./T_perturb/T_perturb/batch_job_script/cytoimmgen/4_run_val_generate_interpolation.sh
```



## Citation

If you use our repository or code in your research, please cite us:

```

```
