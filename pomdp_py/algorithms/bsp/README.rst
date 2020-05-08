Belief Space Planning
=====================

An attempt to implement the BLQR algorithm in the paper:

Reference: `Belief space planning assuming maximum likelihood observations <http://groups.csail.mit.edu/robotics-center/public_papers/Platt10.pdf>`_.

Current status: **Not complete.** The scipy's `SLSQP
<https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html#optimize-minimize-slsqp>`_
does not seem to be able to solve the quadratic program (cannot satisfy the
constraints). Contribution is wanted (either rewrite the whole algorithm or use a different solver).
This algorithm should be used to solve the `light dark domain <https://github.com/h2r/pomdp-py/tree/master/pomdp_problems/light_dark>`_.
