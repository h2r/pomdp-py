# pomdp-py

A package that contains implementation of POMDP, Object-Oriented POMDP, and Abstract POMDP.
Includes solvers, such as POMCP, OO-POMCP. Integrated with [`simple_rl`](https://github.com/david-abel/simple_rl). 

Can be used for POMDP related research, or if you want to formulate a problem as a POMDP as solve it.
Just describe your problem in the `pomdp` format, and use an existing solver to solve it.

There are existing repositories of POMDP implementation in Python. Currently, the additional value of this package
is the object-oriented POMDP implementation, abstract pomdp implementation, and a parser for easy POMDP description.

## Getting started

Currently, `pomdp-py` only supports Python 3.5+ (tested on 3.5 and 3.7).
It has not been tested on Python 2, so it will likely not work.


Install `simple_rl` as follows:
```
git clone git@github.com:zkytony/simple_rl.git
cd simple_rl
git checkout -b kaiyu/pomdp
```

Install `pomdp-py` as a developer
```
git clone git@github.com:zkytony/pomdp-py.git
cd pomdp-py
pip install -e .
```

Test it out with test scripts.

* POMCP for POMDP
```
python test_pomcp_pomdp.py
```

* OOPOMCP for OOPOMDP
```
python test_oopomcp_ooxpomdp.py
```

## Using pomdp-py
TODO: Finish this readme when there are more examples.
