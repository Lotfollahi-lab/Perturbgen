from setuptools import setup

setup(
    name="geneformer",
    version="0.1.0",
    author="Christina Theodoris",
    author_email="christina.theodoris@gladstone.ucsf.edu",
    description="Geneformer is a transformer model pretrained \
                 on a large-scale corpus of ~30 million single \
                 cell transcriptomes to enable context-aware \
                 predictions in settings with limited data in \
                 network biology.",
    packages=["geneformer"],
    python_requires=">=3.10",
    include_package_data=True,
    install_requires=[
        "anndata",
        "datasets",
        "loompy",
        "matplotlib",
        "numpy",
        "packaging",
        "pandas",
        "pyarrow",
        "pytz",
        "ray",
        "scanpy",
        "scikit-learn",
        "scipy",
        "seaborn",
        "setuptools",
        "statsmodels",
        "tdigest",
        "torch",
        "tqdm",
        "transformers",
    ],
)
