"""
Light-Dark Domain
=================

A simple continuous domain. 

Reference: `Belief space planning assuming maximum likelihood observations <http://groups.csail.mit.edu/robotics-center/public_papers/Platt10.pdf>`_.

`Quoting from the original paper on problem description`:

    In the light-dark domain, a robot must localize its position in the plane before approaching the goal. The robot’s ability to localize itself depends upon the amount of light present at its actual position. Light varies as a quadratic function of the horizontal coordinate. Depending upon the goal position, the initial robot position, and the configuration of the light, the robot may need to move away from its ultimate goal in order to localize itself. Figure 1 illustrates the configuration of the light-dark domain used in our experiments. The goal position is at the origin, marked by an X in the figure. The intensity in the figure illustrates the magnitude of the light over the plane. The robot’s initial position is unknown.

.. figure:: https://i.imgur.com/7OYr6Hh.jpg
   :alt: Figure from the paper

   Light-Dark domain
   

**Not yet implemented**
"""
