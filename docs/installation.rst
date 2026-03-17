Installation
============

To get started with PerturbGen, clone the git repository:

.. code:: shell-session

    $ git clone https://github.com/Lotfollahi-lab/Perturbgen.git
    $ cd Perturbgen/

Install Poetry (one-time):
(wanna know what is poetry? have a look at https://python-poetry.org)

.. code:: shell-session

    $ curl -sSL https://install.python-poetry.org | python3 -


Optional: alternative way to install poetry using pipx (https://pipx.pypa.io/stable/installation/)

.. code:: shell-session

    $ pipx install poetry

Create/install the environment and dependencies:

.. code:: shell-session

    $ cd Perturbgen
    $ poetry env use python3.11
    $ poetry install


It's good practice to create a virtual environment to avoid polluting the rest of the system:
Activate the environment

.. code:: shell-session

   $ source "$(poetry env info -p)/bin/activate"

The project contains some jupyter notebooks, which were converted to python files
due to better handling in the repository.
These files end with `_nb.py` and can be converted back to a `.ipynb` file with
`jupytext`:

.. code:: shell-session

   $ jupytext --to ipynb --execute <your_file>_nb.py


PerturbGen is structured as an installable Python package, so installing the dependencies is easy:

.. code:: shell-session

   $ pip install -e .

You can then run the ``perturbgen`` command:

.. code:: shell-session

   $ perturbgen
   Usage: perturbgen [OPTIONS] COMMAND [ARGS]...

   Options:
   --help  Show this message and exit.

   Commands:
   tokenise       Data preprocessing, tokenisation
   train-mask     Training the masking model
   train-decoder  Training the count decoder model
   generate       Load checkpoint and generate predictions
