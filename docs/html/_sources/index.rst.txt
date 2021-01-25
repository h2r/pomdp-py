
pomdp_py Documentation
======================


Overview
--------
`pomdp_py <https://github.com/h2r/pomdp-py>`_ is a **general purpose POMDP library** written in Python and Cython. It features simple and comprehensive interfaces to describe POMDP or MDP problems. Originally written to support POMDP planning research, the interfaces also allow extensions to model-free or model-based learning in (PO)MDPs, multi-agent POMDP planning/learning, and task transfer or transfer learning.

**Why pomdp_py?** It provides a POMDP framework in Python with clean and intuitive interfaces. This makes POMDP-related research or projects accessible to more people. It also helps sharing code and developing a community.

POMDP stands for **P**\ artially **O**\ bservable **M**\ arkov **D**\ ecision **P**\ rocess :cite:`kaelbling1998planning`.

The code is available `on github <https://github.com/h2r/pomdp-py>`_. We welcome contributions to this library in:

(1) Implementation of additional POMDP solvers (see :doc:`existing_solvers`)
(2) Implementation of additional POMDP domains (see :doc:`examples`)
(3) Interfacing with `existing POMDP libraries <other_libraries.html>`_ (majority in other languages).
(4) Extension of `pomdp_py` beyond planning (see :doc:`extensions`).



Getting Started
---------------
* :doc:`installation`
* :doc:`examples` (:doc:`examples.tiger`, :doc:`examples.mos`).
  The Tiger example represents how one could quickly implement a simple POMDP; The Multi-Object Search domain represents how one may structure their project (as a convention) when using `pomdp_py` for more complicated domains.

* :doc:`design_principles`
* :doc:`existing_solvers`
* :doc:`examples.external_solvers`

Further
-------
* :doc:`design_principles`
* :doc:`existing_solvers`
* :doc:`extensions`
* :doc:`other_libraries`
* :doc:`simple_rl_integration`
* :doc:`extensions`

.. toctree::
   :maxdepth: 2
   :caption: Overview
   :hidden:

   installation
   examples
   design_principles
   existing_solvers
   extensions


Citation
--------
If you find this library helpful to your work, please cite the `following paper <https://arxiv.org/pdf/2004.10099.pdf>`_

.. code-block:: bibtex

 @inproceedings{zheng2020pomdp_py,
    title = {pomdp\_py: A Framework to Build and Solve POMDP Problems},
    author = {Zheng, Kaiyu and Tellex, Stefanie},
    booktitle = {ICAPS 2020 Workshop on Planning and Robotics (PlanRob)},
    year = {2020},
    url = {https://icaps20subpages.icaps-conference.org/wp-content/uploads/2020/10/14-PlanRob_2020_paper_3.pdf},
    note = {Arxiv link: "\url{https://arxiv.org/pdf/2004.10099.pdf}"}
 }

API References
--------------
.. toctree::
   :maxdepth: 3
   :caption: API References

   api/modules
   problems/modules

Tools
-----
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. bibliography:: refs.bib
   :filter: docname in docnames
   :style: unsrt

.. image:: images/brown_logo.png
   :target: http://bigai.cs.brown.edu/
   :scale: 25 %
   :alt: Brown University AI
   :align: center
