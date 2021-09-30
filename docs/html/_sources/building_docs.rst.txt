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

Note that when creating the documentation for a problem under :code:`pomdp_problems`,
you may want to re-use the README file on github for the documentation. To do that,
first create a read-me file at :code:`pomdp_problems/<problem>/README.rst` with desirable
content that describes the problem. Then, include this read-me file at the top of the
generated :code:`pomdp_problems.<problem>.rst`, like so:

.. code-block::

   .. include:: ../../../pomdp_problems/<problem>/README.rst


Note on Changelog
-----------------
:doc:`changelog` is generated based on :code:`CHANGELOG.rst` in the repository's root directory.
When the website is constructed, the :code:`_sphinx_src/changelog.rst` is a symbolic
link to :code:`CHANGELOG.rst`, created by

.. code-block::

   cd _sphinx_src
   ln -s ../../CHANGELOG.rst changelog.rst

But because github pages cannot access :code:`../../CHANGELOG.rst`, this symbolic has to be removed
when the site is deployed.
