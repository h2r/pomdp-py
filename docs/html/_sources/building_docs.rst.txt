Building Documentations
=======================

Documentations are based on `Sphinx <https://www.sphinx-doc.org/en/master/>`_.
Here we explain steps to build documentations from source, that is, to generate the documentation web pages.


There are two packages: `pomdp_py` and `pomdp_problems`. Their documentations are built separately.

Building docs for `pomdp_py`
----------------------------

1. Go to the sphinx source directory::

    cd pomdp-py/docs/_sphinx_src

2. Building docs for `pomdp_py`. Run :code:`sphinx-apidoc`::

    sphinx-apidoc -o api/ ../../pomdp_py

   This generates `.rst` files for different modules in the `pomdp_py` package.

   The first argument :code:`api` specifies the output directory for these `.rst` files to be :code:`api`.

   The second argument :code:`../../pomdp_py` specifies the path to the `pomdp_py` package (Note that this should be the package's root path that contains :code:`__init__.py`).

   | Refer to `sphinx-apidoc <https://www.sphinx-doc.org/en/master/man/sphinx-apidoc.html>`_ for more information.



3. Generate web pages::

    make html

   This outputs to :code:`pomdp-py/docs/html`


Building docs for `pomdp_problems`
----------------------------------


1. Go to the sphinx source directory::

    cd pomdp-py/docs/_sphinx_src

2. Building docs for `pomdp_py`. Run :code:`sphinx-apidoc`::

    sphinx-apidoc -o problems/ ../../pomdp_problems

3. Generate web pages::

    make html

   This outputs to :code:`pomdp-py/docs/html`
