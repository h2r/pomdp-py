
pomdp_py Documentation
======================

pomdp_py is a **general purpose POMDP library** written in Python and Cython. It features simple and comprehensive interfaces to describe POMDP or MDP problems. Originally written to support POMDP planning research, the interfaces also allow extensions to model-free or model-based learning in (PO)MDPs, multi-agent POMDP planning/learning, and task transfer or transfer learning.

POMDP stands for **P**\ artially **O**\ bservable **M**\ arkov **D**\ ecision **P**\ rocess :cite:`kaelbling1998planning`.

.. figure:: pomdp.jpg
   :alt: POMDP diagram

   A Diagram for POMDP :math:`\langle S,A,\Omega,T,O,R \rangle`. (**error**:
   change :math:`o\in S` to :math:`o\in\Omega`)
   

Design Principles
#################
1. **Distributions are the fundamental building blocks of a POMDP**.

   Essentially, a POMDP describes the interaction between an `agent` and the
   `environment`.  The interaction is formally encapsulated by a few important
   `generative probability distributions`. The core of pomdp_py is built around
   this understanding. The interfaces in :py:mod:`pomdp_py.framework.basics`
   convey this idea.

   * :class:`~pomdp_py.framework.basics.Distribution` represents :math:`\Pr:X\rightarrow[0,1]`.
   * :class:`~pomdp_py.framework.basics.GenerativeDistribution` should support computing  :math:`argmax_x\Pr(x)` (i.e. MPE) or random sampling :math:`x\sim Pr(x)`.
   * :class:`~pomdp_py.framework.basics.TransitionModel` 


Full API Reference
##################  
.. toctree::
   :maxdepth: 4

   api/modules
   
           
More
----

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. bibliography:: refs.bib
   :style: unsrt

