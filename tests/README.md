# Test output

09/29/2021

Ubuntu 18.04, gcc 7.5.0

```
$ python test_all.py
[1/7] testing conversion to .pomdp file
[testing] test_pomdp_file_conversion
[testing] Running pomdp-solve on generated ./test_tiger.POMDP

** Warning **
        lp_solve reported 2 LPs with numerical instability.
Pass.
[2/7] testing conversion to .pomdpx file
[testing] test_pomdpx_file_conversion
[testing] converting to .pomdpx file
Pass.
[3/7] testing hashing & pickling some objects
Pass.
[4/7] testing particle representation
[5/7] testing sarsop
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
[testing] running computed policy graph(step=2, action=open-right, observation=tiger-left, reward=10)
[testing] running computed policy graph(step=3, action=listen, observation=tiger-left, reward=-1)
[testing] running computed policy graph(step=4, action=listen, observation=tiger-left, reward=-1)
[testing] running computed policy graph(step=5, action=open-right, observation=tiger-right, reward=10)
[testing] running computed policy graph(step=6, action=listen, observation=tiger-left, reward=-1)
[testing] running computed policy graph(step=7, action=listen, observation=tiger-left, reward=-1)
[testing] running computed policy graph(step=8, action=open-right, observation=tiger-right, reward=10)
[testing] running computed policy graph(step=9, action=listen, observation=tiger-left, reward=-1)
Pass.
[6/7] testing pomdp_py.utils.TreeDebugger
Printing tree up to depth 1
_VNodePP(n=4095, v=-20.626)(depth=0)
├─── ₀listen⟶_QNodePP(n=4038, v=-20.626)
│    ├─── ₀tiger-left⟶_VNodePP(n=2004, v=-16.585)(depth=1)
│    │    ├─── ₀listen⟶_QNodePP(n=1816, v=-16.585)
│    │    ├─── ₁open-left⟶_QNodePP(n=20, v=-131.071)
│    │    └─── ₂open-right⟶_QNodePP(n=168, v=-48.294)
│    └─── ₁tiger-right⟶_VNodePP(n=2032, v=-17.214)(depth=1)
│         ├─── ₀listen⟶_QNodePP(n=1868, v=-17.214)
│         ├─── ₁open-left⟶_QNodePP(n=141, v=-51.569)
│         └─── ₂open-right⟶_QNodePP(n=23, v=-120.513)
├─── ₁open-left⟶_QNodePP(n=27, v=-125.426)
│    ├─── ₀tiger-left⟶_VNodePP(n=16, v=-47.265)(depth=1)
│    │    ├─── ₀listen⟶_QNodePP(n=9, v=-47.265)
│    │    ├─── ₁open-left⟶_QNodePP(n=3, v=-156.101)
│    │    └─── ₂open-right⟶_QNodePP(n=4, v=-142.408)
│    └─── ₁tiger-right⟶_VNodePP(n=9, v=-41.256)(depth=1)
│         ├─── ₀listen⟶_QNodePP(n=4, v=-41.256)
│         ├─── ₁open-left⟶_QNodePP(n=2, v=-138.689)
│         └─── ₂open-right⟶_QNodePP(n=3, v=-84.601)
└─── ₂open-right⟶_QNodePP(n=30, v=-120.933)
     ├─── ₀tiger-left⟶_VNodePP(n=18, v=-51.758)(depth=1)
     │    ├─── ₀listen⟶_QNodePP(n=10, v=-51.758)
     │    ├─── ₁open-left⟶_QNodePP(n=4, v=-109.814)
     │    └─── ₂open-right⟶_QNodePP(n=4, v=-110.144)
     └─── ₁tiger-right⟶_VNodePP(n=10, v=-47.396)(depth=1)
          ├─── ₀listen⟶_QNodePP(n=4, v=-47.396)
          ├─── ₁open-left⟶_QNodePP(n=3, v=-86.168)
          └─── ₂open-right⟶_QNodePP(n=3, v=-87.910)
==== Step 1 ====
True state: tiger-left
Belief: [(TigerState(tiger-right), 0.5), (TigerState(tiger-left), 0.5)]
Action: listen
Reward: -1
>> Observation: tiger-left
Num sims: 4096
Plan time: 0.12775
==== Step 2 ====
True state: tiger-left
Belief: [(TigerState(tiger-left), 0.85), (TigerState(tiger-right), 0.15)]
Action: listen
Reward: -1
>> Observation: tiger-left
Num sims: 4096
Plan time: 0.13018
==== Step 3 ====
True state: tiger-left
Belief: [(TigerState(tiger-left), 0.9697986575573173), (TigerState(tiger-right), 0.03020134244268276)]
Action: listen
Reward: -1
>> Observation: tiger-left
Num sims: 4096
Plan time: 0.15164
[7/7] testing vi_pruning (pomdp-solve)
input file   : ./temp-pomdp.pomdp
 //****************\\
||   pomdp-solve    ||
||     v. 5.4       ||
 \\****************//
      PID=28755
- - - - - - - - - - - - - - - - - - - -
time_limit = 0
mcgs_prune_freq = 100
verbose = context
stdout =
inc_prune = normal
history_length = 0
prune_epsilon = 0.000000
save_all = false
o = temp-pomdp
fg_save = false
enum_purge = normal_prune
fg_type = initial
fg_epsilon = 0.000000
mcgs_traj_iter_count = 1
lp_epsilon = 0.000000
end_epsilon = 0.000000
start_epsilon = 0.000000
dom_check = false
stop_delta = 0.000000
q_purge = normal_prune
pomdp = ./temp-pomdp.pomdp
mcgs_num_traj = 1000
stop_criteria = weak
method = incprune
memory_limit = 0
alg_rand = 0
terminal_values =
save_penultimate = false
epsilon = 0.000000
rand_seed =
discount = -1.000000
fg_points = 10000
fg_purge = normal_prune
proj_purge = normal_prune
mcgs_traj_length = 100
history_delta = 0
f =
epsilon_adjust = 0.000000
prune_rand = 0
vi_variation = normal
horizon = 100
stat_summary = false
max_soln_size = 0.000000
witness_points = false
- - - - - - - - - - - - - - - - - - - -
[Initializing POMDP ... done.]
[Initial policy has 1 vectors.]
++++++++++++++++++++++++++++++++++++++++
Epoch: 1...3 vectors in 0.00 secs. (0.00 total) (err=inf)
Epoch: 2...5 vectors in 0.00 secs. (0.00 total) (err=inf)
Epoch: 3...9 vectors in 0.00 secs. (0.00 total) (err=inf)
Epoch: 4...7 vectors in 0.00 secs. (0.00 total) (err=inf)
Epoch: 5...13 vectors in 0.00 secs. (0.00 total) (err=inf)
Epoch: 6...15 vectors in 0.00 secs. (0.00 total) (err=inf)
Epoch: 7...19 vectors in 0.00 secs. (0.00 total) (err=inf)
Epoch: 8...25 vectors in 0.00 secs. (0.00 total) (err=inf)
Epoch: 9...27 vectors in 0.01 secs. (0.01 total) (err=inf)
Epoch: 10...27 vectors in 0.00 secs. (0.01 total) (err=inf)
Epoch: 11...37 vectors in 0.01 secs. (0.02 total) (err=inf)
Epoch: 12...35 vectors in 0.02 secs. (0.04 total) (err=inf)
Epoch: 13...39 vectors in 0.01 secs. (0.05 total) (err=inf)
Epoch: 14...47 vectors in 0.02 secs. (0.07 total) (err=inf)
Epoch: 15...47 vectors in 0.03 secs. (0.10 total) (err=inf)
Epoch: 16...47 vectors in 0.02 secs. (0.12 total) (err=inf)
Epoch: 17...53 vectors in 0.03 secs. (0.15 total) (err=inf)
Epoch: 18...51 vectors in 0.04 secs. (0.19 total) (err=inf)
Epoch: 19...57 vectors in 0.03 secs. (0.22 total) (err=inf)
Epoch: 20...59 vectors in 0.05 secs. (0.27 total) (err=inf)
Epoch: 21...61 vectors in 0.05 secs. (0.32 total) (err=inf)
Epoch: 22...61 vectors in 0.04 secs. (0.36 total) (err=inf)
Epoch: 23...61 vectors in 0.05 secs. (0.41 total) (err=inf)
Epoch: 24...61 vectors in 0.06 secs. (0.47 total) (err=inf)
Epoch: 25...63 vectors in 0.05 secs. (0.52 total) (err=inf)
Epoch: 26...65 vectors in 0.05 secs. (0.57 total) (err=inf)
Epoch: 27...63 vectors in 0.07 secs. (0.64 total) (err=inf)
Epoch: 28...65 vectors in 0.05 secs. (0.69 total) (err=inf)
Epoch: 29...66 vectors in 0.06 secs. (0.75 total) (err=inf)
Epoch: 30...60 vectors in 0.05 secs. (0.80 total) (err=inf)
Epoch: 31...59 vectors in 0.04 secs. (0.84 total) (err=inf)
Epoch: 32...58 vectors in 0.04 secs. (0.88 total) (err=inf)
Epoch: 33...49 vectors in 0.03 secs. (0.91 total) (err=inf)
Epoch: 34...51 vectors in 0.02 secs. (0.93 total) (err=inf)
Epoch: 35...51 vectors in 0.03 secs. (0.96 total) (err=inf)
Epoch: 36...52 vectors in 0.03 secs. (0.99 total) (err=inf)
Epoch: 37...51 vectors in 0.02 secs. (1.01 total) (err=inf)
Epoch: 38...46 vectors in 0.03 secs. (1.04 total) (err=inf)
Epoch: 39...47 vectors in 0.02 secs. (1.06 total) (err=inf)
Epoch: 40...47 vectors in 0.02 secs. (1.08 total) (err=inf)
Epoch: 41...41 vectors in 0.02 secs. (1.10 total) (err=inf)
Epoch: 42...46 vectors in 0.02 secs. (1.12 total) (err=inf)
Epoch: 43...39 vectors in 0.01 secs. (1.13 total) (err=inf)
Epoch: 44...37 vectors in 0.01 secs. (1.14 total) (err=inf)
Epoch: 45...39 vectors in 0.02 secs. (1.16 total) (err=inf)
Epoch: 46...41 vectors in 0.01 secs. (1.17 total) (err=inf)
Epoch: 47...33 vectors in 0.01 secs. (1.18 total) (err=inf)
Epoch: 48...31 vectors in 0.01 secs. (1.19 total) (err=inf)
Epoch: 49...33 vectors in 0.01 secs. (1.20 total) (err=inf)
Epoch: 50...31 vectors in 0.00 secs. (1.20 total) (err=inf)
Epoch: 51...29 vectors in 0.01 secs. (1.21 total) (err=inf)
Epoch: 52...25 vectors in 0.00 secs. (1.21 total) (err=inf)
Epoch: 53...27 vectors in 0.01 secs. (1.22 total) (err=inf)
Epoch: 54...23 vectors in 0.00 secs. (1.22 total) (err=inf)
Epoch: 55...21 vectors in 0.01 secs. (1.23 total) (err=inf)
Epoch: 56...21 vectors in 0.00 secs. (1.23 total) (err=inf)
Epoch: 57...27 vectors in 0.00 secs. (1.23 total) (err=inf)
Epoch: 58...21 vectors in 0.01 secs. (1.24 total) (err=inf)
Epoch: 59...19 vectors in 0.00 secs. (1.24 total) (err=inf)
Epoch: 60...25 vectors in 0.00 secs. (1.24 total) (err=inf)
Epoch: 61...17 vectors in 0.00 secs. (1.24 total) (err=inf)
Epoch: 62...15 vectors in 0.01 secs. (1.25 total) (err=inf)
Epoch: 63...15 vectors in 0.00 secs. (1.25 total) (err=inf)
Epoch: 64...13 vectors in 0.00 secs. (1.25 total) (err=inf)
Epoch: 65...13 vectors in 0.00 secs. (1.25 total) (err=inf)
Epoch: 66...11 vectors in 0.00 secs. (1.25 total) (err=inf)
Epoch: 67...11 vectors in 0.00 secs. (1.25 total) (err=inf)
Epoch: 68...13 vectors in 0.00 secs. (1.25 total) (err=inf)
Epoch: 69...11 vectors in 0.00 secs. (1.25 total) (err=inf)
Epoch: 70...11 vectors in 0.00 secs. (1.25 total) (err=inf)
Epoch: 71...9 vectors in 0.01 secs. (1.26 total) (err=inf)
Epoch: 72...9 vectors in 0.00 secs. (1.26 total) (err=inf)
Epoch: 73...9 vectors in 0.00 secs. (1.26 total) (err=inf)
Epoch: 74...9 vectors in 0.00 secs. (1.26 total) (err=inf)
Epoch: 75...9 vectors in 0.00 secs. (1.26 total) (err=inf)
Epoch: 76...9 vectors in 0.00 secs. (1.26 total) (err=inf)
Epoch: 77...9 vectors in 0.00 secs. (1.26 total) (err=inf)
Epoch: 78...9 vectors in 0.00 secs. (1.26 total) (err=inf)
Epoch: 79...9 vectors in 0.00 secs. (1.26 total) (err=inf)
Epoch: 80...9 vectors in 0.00 secs. (1.26 total) (err=inf)
Epoch: 81...9 vectors in 0.00 secs. (1.26 total) (err=inf)
Epoch: 82...9 vectors in 0.00 secs. (1.26 total) (err=inf)
Epoch: 83...9 vectors in 0.01 secs. (1.27 total) (err=inf)
Epoch: 84...9 vectors in 0.00 secs. (1.27 total) (err=inf)
Epoch: 85...9 vectors in 0.00 secs. (1.27 total) (err=inf)
Epoch: 86...9 vectors in 0.00 secs. (1.27 total) (err=inf)
Epoch: 87...9 vectors in 0.00 secs. (1.27 total) (err=inf)
Epoch: 88...9 vectors in 0.00 secs. (1.27 total) (err=inf)
Epoch: 89...9 vectors in 0.00 secs. (1.27 total) (err=inf)
Epoch: 90...9 vectors in 0.00 secs. (1.27 total) (err=inf)
Epoch: 91...9 vectors in 0.00 secs. (1.27 total) (err=inf)
Epoch: 92...9 vectors in 0.00 secs. (1.27 total) (err=inf)
Epoch: 93...9 vectors in 0.00 secs. (1.27 total) (err=inf)
Epoch: 94...9 vectors in 0.00 secs. (1.27 total) (err=inf)
Epoch: 95...9 vectors in 0.01 secs. (1.28 total) (err=inf)
Epoch: 96...9 vectors in 0.00 secs. (1.28 total) (err=inf)
Epoch: 97...9 vectors in 0.00 secs. (1.28 total) (err=inf)
Epoch: 98...9 vectors in 0.00 secs. (1.28 total) (err=inf)
Epoch: 99...9 vectors in 0.00 secs. (1.28 total) (err=inf)
Epoch: 100...9 vectors in 0.00 secs. (1.28 total) (err=inf)
++++++++++++++++++++++++++++++++++++++++
Solution found.  See file:
	temp-pomdp.alpha
	temp-pomdp.pg
++++++++++++++++++++++++++++++++++++++++
User time = 0 hrs., 0 mins, 1.28 secs. (= 1.28 secs)
System time = 0 hrs., 0 mins, 0.00 secs. (= 0.00 secs)
Total execution time = 0 hrs., 0 mins, 1.28 secs. (= 1.28 secs)

** Warning **
        lp_solve reported 2 LPs with numerical instability.
[testing] test_vi_pruning
[testing] solving the tiger problem...
[testing] simulating computed policy graph(step=0, action=listen, observation=tiger-right, reward=-1)
[testing] simulating computed policy graph(step=1, action=listen, observation=tiger-left, reward=-1)
[testing] simulating computed policy graph(step=2, action=listen, observation=tiger-left, reward=-1)
[testing] simulating computed policy graph(step=3, action=listen, observation=tiger-left, reward=-1)
[testing] simulating computed policy graph(step=4, action=open-right, observation=tiger-left, reward=10)
[testing] simulating computed policy graph(step=5, action=listen, observation=tiger-left, reward=-1)
[testing] simulating computed policy graph(step=6, action=listen, observation=tiger-left, reward=-1)
[testing] simulating computed policy graph(step=7, action=open-right, observation=tiger-right, reward=10)
[testing] simulating computed policy graph(step=8, action=listen, observation=tiger-left, reward=-1)
[testing] simulating computed policy graph(step=9, action=listen, observation=tiger-left, reward=-1)
Pass.
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
