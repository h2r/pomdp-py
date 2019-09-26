import random
from maze1d_pomdp import unittest as unittest_pomdp
from maze1d_hierarchical import unittest as unittest_hierarchical
import numpy as np


worldlen = 30
num_segments = 5

results_pomdp = []
results_search = []
results_backtrack = []

try:

    for i in range(30):
        print()
        print()
        print("~~~~~~~~~~~~~~~~~~~~Case %d~~~~~~~~~~~~~~~~~~~~" % i)
        world = ["."] * worldlen
        robot_pose = random.randint(0,worldlen-1)
        target_pose = random.randint(0,worldlen-1)
        while robot_pose == target_pose:
            target_pose = random.randint(0,worldlen-1)
        world[robot_pose] = "R"
        world[target_pose] = "T"
        total_time, rewards = unittest_pomdp(world)  # normal pomdp
        results_pomdp.append((total_time, sum(rewards)))
        total_time, rewards = unittest_hierarchical(world, num_segments, True)   # search
        results_search.append((total_time, sum(rewards)))
        total_time, rewards = unittest_hierarchical(world, num_segments, False)  # backtrack
        results_backtrack.append((total_time, sum(rewards)))

except Exception as ex:
    print("error!")
    raise ex
finally:
    # Average and std
    print("Normal POMDP:")
    time = [r[0] for r in results_pomdp]
    rewards = [r[1] for r in results_pomdp]
    print("  Average time: %.3f" % (np.mean(time)))
    print("      std time: %.3f" % (np.std(time)))
    print("  Average reward: %.3f" % (np.mean(rewards)))
    print("      std reward: %.3f" % (np.std(rewards)))

    print("Abstract POMDP (Search+Backtrack):")
    time = [r[0] for r in results_search]
    rewards = [r[1] for r in results_search]
    print("  Average time: %.3f" % (np.mean(time)))
    print("      std time: %.3f" % (np.std(time)))
    print("  Average reward: %.3f" % (np.mean(rewards)))
    print("      std reward: %.3f" % (np.std(rewards)))

    print("Abstract POMDP (Backtrack):")
    time = [r[0] for r in results_backtrack]
    rewards = [r[1] for r in results_backtrack]
    print("  Average time: %.3f" % (np.mean(time)))
    print("      std time: %.3f" % (np.std(time)))
    print("  Average reward: %.3f" % (np.mean(rewards)))
    print("      std reward: %.3f" % (np.std(rewards)))
