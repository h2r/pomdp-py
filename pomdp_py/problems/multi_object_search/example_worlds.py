"""This file has some examples of world string."""
import random

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
rx.T...
.x.....
...xx..
.......
.xxx.T.
.xxx...
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
..xxxTxxxxxxxxx..
..xxxxxxxxxxxxx..
..xxxx...xxxxxx..
..xxxx..xx.xxxx..
..xxxx..r.Txxxx..
..xxxx..xx.xxxx..
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

def random_world(width, length, num_obj, num_obstacles,
                 robot_char="r"):
    worldstr = [[ "." for i in range(width)] for j in range(length)]
    # First place obstacles
    num_obstacles_placed = 0
    while num_obstacles_placed < num_obstacles:
        x = random.randrange(0, width)
        y = random.randrange(0, length)
        if worldstr[y][x] == ".":
            worldstr[y][x] = "x"
            num_obstacles_placed += 1
            
    num_obj_placed = 0
    while num_obj_placed < num_obj:
        x = random.randrange(0, width)
        y = random.randrange(0, length)
        if worldstr[y][x] == ".":
            worldstr[y][x] = "T"
            num_obj_placed += 1

    # Finally place the robot
    while True:
        x = random.randrange(0, width)
        y = random.randrange(0, length)
        if worldstr[y][x] == ".":
            worldstr[y][x] = robot_char
            break

    # Create the string.
    finalstr = []
    for row_chars in worldstr:
        finalstr.append("".join(row_chars))
    finalstr = "\n".join(finalstr)
    return finalstr, robot_char
