.. PerturbGen documentation master file, created by
   sphinx-quickstart on Thu Aug 14 16:14:56 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

PerturbGen
**********

PerturbGen is a scientific Python library for modeling, generating, and analyzing cellular perturbation effects from high-dimensional biological data. It is designed for researchers and developers working with perturbation experiments (e.g. genetic, chemical, or CRISPR-based perturbations) who want reproducible, model-driven ways to simulate and interpret perturbational responses.

Conceptual overview
===================

At a high level, a typical PerturbGen workflow looks like:

- Prepare perturbation-aware biological data (e.g. single-cell expression)

- Configure or load a perturbation model

- Apply perturbations in silico to generate predicted responses

- Analyze, compare, or visualize the resulting perturbation effects

PerturbGen does not aim to be a general-purpose machine learning framework. Instead, it provides abstractions that encode biological perturbations as first-class objects, making downstream analyses easier to reason about and reproduce.

See Also
--------

- `GitHub repo <https://github.com/Lotfollahi-lab/Perturbgen>`__
- `Hugging Face repo <https://huggingface.co/lotfollahi-lab/PerturbGen/tree/main>`__


.. toctree::
      
   :maxdepth: 2

   installation
   data
   tutorial

   API <apidoc/perturbgen/modules>
