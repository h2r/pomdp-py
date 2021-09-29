# Test output

09/29/2021

Ubuntu 18.04, gcc 7.5.0

```
$ python test_all.py
[1/6] testing conversion to .pomdp file
[testing] test_pomdp_file_conversion
[testing] Running pomdp-solve on generated ./test_tiger.POMDP

** Warning **
        lp_solve reported 2 LPs with numerical instability.
Pass.
[2/6] testing conversion to .pomdpx file
[testing] test_pomdpx_file_conversion
[testing] converting to .pomdpx file
input file   : ./temp-pomdp.pomdp
Pass.
[3/6] testing particle representation
[4/6] testing sarsop
[testing] test_sarsop
[testing] solving the tiger problem...

Loading the model ...
  input file   : ./temp-pomdp.pomdp
  loading time : 0.00s

SARSOP initializing ...
  initialization time : 0.00s

-------------------------------------------------------------------------------
 Time   |#Trial |#Backup |LBound    |UBound    |Precision  |#Alphas |#Beliefs
-------------------------------------------------------------------------------
 0       0       0        -20        92.8205    112.821     3        1
 0       2       51       -6.2981    63.1395    69.4376     7        16
 0       4       103      0.149651   52.2764    52.1267     9        21
 0       6       151      6.19248    42.0546    35.8621     9        21
 0       8       200      10.3563    35.232     24.8756     12       21
 0       11      250      14.0433    29.5471    15.5037     6        21
 0       14      300      16.545     25.0926    8.54758     10       21
 0       17      350      18.2281    21.8162    3.5882      14       21
 0       18      400      18.7451    20.9384    2.19328     8        21
 0       21      465      19.1109    20.0218    0.910954    5        21
 0       22      500      19.2369    19.7071    0.470218    11       21
 0       24      550      19.3036    19.5405    0.236865    6        21
 0       25      600      19.3369    19.4574    0.120445    13       21
 0       27      669      19.3579    19.4049    0.0469304   5        21
 0       28      713      19.3643    19.389     0.0247389   5        21
 0       29      757      19.3676    19.3807    0.0130409   5        21
 0       30      801      19.3694    19.3763    0.00687438  5        21
 0       31      850      19.3704    19.3739    0.00351432  10       21
 0       32      900      19.3709    19.3725    0.00155165  5        21
 0       33      950      19.3712    19.3719    0.000736099 11       21
 0       35      1021     19.3713    19.3716    0.000279811 5        21
 0       36      1065     19.3713    19.3715    0.0001475   5        21
 0       37      1109     19.3713    19.3714    7.77531e-05 5        21
 0       38      1153     19.3714    19.3714    4.09868e-05 5        21
 0       39      1200     19.3714    19.3714    2.09533e-05 8        21
 0       40      1250     19.3714    19.3714    9.84548e-06 14       21
 0       41      1300     19.3714    19.3714    4.71002e-06 9        21
 0       42      1350     19.3714    19.3714    2.25933e-06 15       21
 0       44      1417     19.3714    19.3714    8.7943e-07  5        21
 0       44      1417     19.3714    19.3714    8.7943e-07  5        21
-------------------------------------------------------------------------------

SARSOP finishing ...
  target precision reached
  target precision  : 0.000001
  precision reached : 0.000001

-------------------------------------------------------------------------------
 Time   |#Trial |#Backup |LBound    |UBound    |Precision  |#Alphas |#Beliefs
-------------------------------------------------------------------------------
 0       44      1417     19.3714    19.3714    8.7943e-07  5        21
-------------------------------------------------------------------------------

Writing out policy ...
  output file : temp-pomdp.policy

[testing] running computed policy graph(step=0, action=listen, observation=tiger-left, reward=-1)
[testing] running computed policy graph(step=1, action=listen, observation=tiger-left, reward=-1)
[testing] running computed policy graph(step=2, action=open-right, observation=tiger-right, reward=10)
[testing] running computed policy graph(step=3, action=listen, observation=tiger-left, reward=-1)
[testing] running computed policy graph(step=4, action=listen, observation=tiger-left, reward=-1)
[testing] running computed policy graph(step=5, action=open-right, observation=tiger-right, reward=10)
[testing] running computed policy graph(step=6, action=listen, observation=tiger-left, reward=-1)
[testing] running computed policy graph(step=7, action=listen, observation=tiger-left, reward=-1)
[testing] running computed policy graph(step=8, action=open-right, observation=tiger-right, reward=10)
[testing] running computed policy graph(step=9, action=listen, observation=tiger-left, reward=-1)
Pass.
[5/6] testing pomdp_py.utils.TreeDebugger
Printing tree up to depth 1
_VNodePP(n=4095, v=-18.663)(depth=0)
├─── ₀listen⟶_QNodePP(n=4044, v=-18.663)
│    ├─── ₀tiger-left⟶_VNodePP(n=1955, v=-15.605)(depth=1)
│    │    ├─── ₀listen⟶_QNodePP(n=1864, v=-15.605)
│    │    ├─── ₁open-left⟶_QNodePP(n=17, v=-140.008)
│    │    └─── ₂open-right⟶_QNodePP(n=74, v=-67.323)
│    └─── ₁tiger-right⟶_VNodePP(n=2087, v=-15.605)(depth=1)
│         ├─── ₀listen⟶_QNodePP(n=2001, v=-15.605)
│         ├─── ₁open-left⟶_QNodePP(n=63, v=-74.215)
│         └─── ₂open-right⟶_QNodePP(n=23, v=-120.490)
├─── ₁open-left⟶_QNodePP(n=26, v=-132.658)
│    ├─── ₀tiger-left⟶_VNodePP(n=13, v=-61.163)(depth=1)
│    │    ├─── ₀listen⟶_QNodePP(n=10, v=-61.163)
│    │    ├─── ₁open-left⟶_QNodePP(n=2, v=-183.225)
│    └─── ₁tiger-right⟶_VNodePP(n=11, v=-68.165)(depth=1)
│         ├─── ₀listen⟶_QNodePP(n=5, v=-68.165)
│         ├─── ₁open-left⟶_QNodePP(n=2, v=-193.551)
│         └─── ₂open-right⟶_QNodePP(n=4, v=-112.495)
└─── ₂open-right⟶_QNodePP(n=25, v=-129.581)
     ├─── ₀tiger-left⟶_VNodePP(n=10, v=-86.844)(depth=1)
     │    ├─── ₀listen⟶_QNodePP(n=4, v=-86.844)
     │    ├─── ₁open-left⟶_QNodePP(n=2, v=-146.526)
     │    └─── ₂open-right⟶_QNodePP(n=4, v=-90.289)
     └─── ₁tiger-right⟶_VNodePP(n=13, v=-69.384)(depth=1)
          ├─── ₀listen⟶_QNodePP(n=9, v=-69.384)
          ├─── ₁open-left⟶_QNodePP(n=2, v=-180.750)
```
## Testing external solvers
The purpose of some tests is to test external solvers using pomdp_py.
These tests require downloading and compiling those solvers.

### Download pomdp_solve
```
wget https://www.pomdp.org/code/pomdp-solve-5.4.tar.gz
```
Then, build the package
according to [Installing from Source section in the documentation](https://www.pomdp.org/code/).


### Download sarsop

Follow the instructions on its [github repo](https://github.com/AdaCompNUS/sarsop).
