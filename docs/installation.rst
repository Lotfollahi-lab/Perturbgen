Installation
============

To get started with Cytomeister, clone the git repository:

.. code:: shell-session

    $ git clone git@github.com:Lotfollahi-lab/T_perturb.git
    $ cd T_perturb/

It's good practice to create a virtual environment to avoid polluting the rest of the system:

.. code:: shell-session

   $ python3 -m venv .venv
   $ source .venv/bin/activate

Cytomeister is structured as an installable Python package, so installing the dependencies is easy:

.. code:: shell-session

   $ pip install -e .

You can then run the ``cytomeister`` command:

.. code:: shell-session

   $ cytomeister
   Usage: cytomeister [OPTIONS] COMMAND [ARGS]...

   Options:
   --help  Show this message and exit.

   Commands:
   tokenise       Data preprocessing, tokenisation
   train-mask     Training the masking model
   train-decoder  Training the count decoder model
   generate       Load checkpoint and generate predictions
