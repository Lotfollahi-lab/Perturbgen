Data
====

Access is provided to ``.h5ad`` files selected for the manuscript as well as the full pretraining corpus *(approximately 1.7 TB)*.

The data is hosted in an S3-compatible bucket at the Wellcome Sanger Institute. Downloading the data requires the AWS CLI to be installed:
https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html

You can browse the available folders before downloading or download a single files using:
https://perturbgen.cog.sanger.ac.uk/data.html


Manuscript data
---------------

To download the dataset used in the manuscript:

.. code-block:: bash

   aws --endpoint-url https://cog.sanger.ac.uk --no-sign-request \
     s3 cp s3://perturbgen/Manuscript Manuscript --recursive

That will create a folder named `Manuscript` in your current working directory containing all the relevant `.h5ad` files.

Pretrain Corpus data
--------------------

To download the full pretraining corpus (~2 TB):

.. code-block:: bash

   aws --endpoint-url https://cog.sanger.ac.uk --no-sign-request \
     s3 cp s3://perturbgen/PretrainCorpus PretrainCorpus --recursive

That will create a folder named `PretrainCorpus` in your current working directory containing all the relevant folders with `.h5ad` files.

.. warning:: Downloading the full pretraining corpus may take a significant amount of time and disk space.
