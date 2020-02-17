:orphan:

Other POMDP Libraries
*********************

There are a number of other libraries; Feel free to check them out and feel the differences. We think our `pomdp_py` library has a simpler and more extensible set of interfaces than these peer libraries. Our implementation in Python and Cython makes the library easy for use in a research project, or a ROS robot system (when `ROS 2 <https://index.ros.org/doc/ros2/>`_ supports Python 3; `ROS <http://wiki.ros.org/Documentation>`_ doesn't support Python 3 but I have worked around the communication between ROS and `pomdp_py` in some other way; The code base is all in Python, which accelerates development.)

* `POMDPy <https://github.com/pemami4911/POMDPy>`_ is written in Python. This library features implementations of advanced Value Iteration algorithms and an implementation of POMCP. The documentation of interfaces is not sufficient, which appear to be more convoluted than `pomdp_py`. A potential connection between the two could be possible if POMDPy provides a clearer documentation of its POMDP interfaces.
|
* `APPL <https://bigbird.comp.nus.edu.sg/pmwiki/farm/appl/>`_ (Approximate POMDP Planning Software) is a C++ library for approximate POMDP planning. It uses `POMDPX <https://bigbird.comp.nus.edu.sg/pmwiki/farm/appl/index.php?n=Main.PomdpXDocumentation>`_ file format to parse POMDPs. This library contains implementation of state-of-the-art planning algorithms such as SARSOP and DESPOT. It would be great to interface `pomdp_py` with this library so that one can make use of these algorithms in Python.
|
* `POMDPs.jl <https://github.com/JuliaPOMDP/POMDPs.jl>`_ is written in Julia, under active development.
  `Julia <https://en.wikipedia.org/wiki/Julia_(programming_language)>`_ is known to be suited for high-performance numerical analysis. Besides POMCP, this library also has an `implementation of DESPOT <https://github.com/JuliaPOMDP/DESPOT.jl>`_, and `SARSOP <https://github.com/JuliaPOMDP/SARSOP.jl>`_. The libray supports porting Python POMDP code into Julia code. Connection between interfaces in `pomdp_py` and `POMDPs.jl` could potentially be done via this Python porting.
|
* `AI-Toolbox <https://github.com/Svalorzen/AI-Toolbox>`_ is written in C++. This library contains a number of algorithms for planning and reinforcement learning in MDP or POMDP problems. It has Python 2 and 3 bindings in specific cases.
|
* `simple_rl <https://github.com/david-abel/simple_rl>`_ is written in Python. It focuses on MDP reinforcement learing algorithms and contains many tasks implementations. It does have a POMDP interface and an implementation of BeliefMDP.    
