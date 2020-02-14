# This file has some examples of world string.

############# Example Worlds ###########
# See env.py:interpret for definition of
# the format

world0 = (
"""
rx...
.x.xT
.....
""", "r")

world1 = (
"""
rx.....
.x..T..
...xx..
.T.....
.xxx...
.xxx.T.
.......
""", "r")

# Used to test the shape of the sensor
world2 = (
"""
.................
.................
..xxxxxxxxxxxxx..
..xxxxxxxxxxxxx..
..xxxxxxxxxxxxx..
..xxxxxxxxxxxxx..
..xxxxxxxxxxxxx..
..xxxxxxxxxxxxx..
..xxxxxxxxxxxxx..
..xxxxxxxxTxxxx..
..xxxxxxrxTxxxx..
..xxxxxxxxTxxxx..
..xxxxxxxxxxxxx..
..xxxxxxxxxxxxx..
..xxxxxxxxxxxxx..
..xxxxxxxxxxxxx..
..xxxxxxxxxxxxx..
..xxxxxxxxxxxxx..
..xxxxxxxxxxxxx..
.................
.................
""", "r")    

# Used to test sensor occlusion
world3 = (
"""
.................
.................
..xxxxxxxxxxxxx..
..xxxxxxxxxxxxx..
..xxxxxxxxxxxxx..
..xxxxxxxxxxxxx..
..xxxxxxxxxxxxx..
..xxxxxxxxxxxxx..
..xxxx...xxxxxx..
..xxxx..xxTxxxx..
..xxxx..rTTxxxx..
..xxxx..xxTxxxx..
..xxxxxx..xxxxx..
..xxxxTx..xxxxx..
..xxxxxxxxxxxxx..
..xxxxxxxxxxxxx..
..xxxxxxxxxxxxx..
..xxxxxxxxxxxxx..
..xxxxxxxxxxxxx..
.................
.................
""", "r")
