# Test output

01/07/2021

Ubuntu 18.04, gcc 7.5.0

```
[testing] test_pomdp_file_conversion
[testing] Running pomdp-solve on generated ./test_tiger.POMDP

** Warning **
        lp_solve reported 2 LPs with numerical instability.
Pass.
[testing] test_pomdpx_file_conversion
[testing] converting to .pomdpx file
input file   : ./temp-pomdp.pomdp
Pass.

... (solver output)
++++++++++++++++++++++++++++++++++++++++
User time = 0 hrs., 0 mins, 2.55 secs. (= 2.55 secs)
System time = 0 hrs., 0 mins, 0.01 secs. (= 0.01 secs)
Total execution time = 0 hrs., 0 mins, 2.56 secs. (= 2.56 secs)

** Warning **
        lp_solve reported 2 LPs with numerical instability.
[testing] running tiger simulation with computed policy graph (step=0)
[testing] running tiger simulation with computed policy graph (step=1)
[testing] running tiger simulation with computed policy graph (step=2)
[testing] running tiger simulation with computed policy graph (step=3)
[testing] running tiger simulation with computed policy graph (step=4)
[testing] running tiger simulation with computed policy graph (step=5)
[testing] running tiger simulation with computed policy graph (step=6)
[testing] running tiger simulation with computed policy graph (step=7)
[testing] running tiger simulation with computed policy graph (step=8)
[testing] running tiger simulation with computed policy graph (step=9)
Pass.
```


## Download pomdp_solve

```
wget https://www.pomdp.org/code/pomdp-solve-5.4.tar.gz
```
Then, build the package
according to [Installing from Source section in the documentation](https://www.pomdp.org/code/).


## Download sarsop

Follow the instructions on its [github repo](https://github.com/AdaCompNUS/sarsop).
