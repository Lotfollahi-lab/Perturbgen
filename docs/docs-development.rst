Documentation
=============

The PerturbGen documentation is written with `Sphinx`_ and `reStructuredText`_.

.. _Sphinx: https://www.sphinx-doc.org/
.. _reStructuredText: https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html

Building
--------

To build the docs, first follow the :doc:`/installation` instructions, then install the documentation-specific dependencies:

.. code:: shell-session

   $ poetry install --with=docs

Then, to build the docs:

.. code:: shell-session

   $ cd docs/
   $ make html

The built HTML will be in the ``docs/_build/html`` directory.

.. note::

   You should not need to run ``sphinx-apidoc`` manually -- it is run automatically when needed.

Live-updating
-------------

You can use `sphinx-autobuild`_ to run a web server and automatically rebuild the documentation when you make changes:

.. _sphinx-autobuild: https://github.com/sphinx-doc/sphinx-autobuild

.. code:: shell-session

   $ make livehtml

Then the docs will be available at http://localhost:8000.

If you need to configure the host/port (e.g. if your browser is not on the machine you're developing the docs on), you can specify options to sphinx-autobuild like so:

.. code:: shell-session

   $ make livehtml O='--host=my-hostname --port=1234'
