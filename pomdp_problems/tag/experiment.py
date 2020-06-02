"""Simple experiment to get mean"""

from pomdp_problems.tag.problem import *
import numpy as np

def trial(worldstr, **kwargs):
    grid_map = GridMap.from_str(worldstr)
    free_cells = grid_map.free_cells()
    init_robot_position = random.sample(free_cells, 1)[0]
    init_target_position = random.sample(free_cells, 1)[0]
    
    problem = TagProblem(init_robot_position,
                         init_target_position,
                         grid_map, **kwargs)
    discounted_reward = solve(problem,
                              max_depth=15,
                              discount_factor=0.95,
                              planning_time=.7,
                              exploration_const=10,
                              visualize=True,
                              max_time=120,
                              max_steps=500)
    return discounted_reward

def main():
    all_rewards = []
    try:
        for i in range(100):
            dr = trial(world0[0],
                       pr_stay=0.5,
                       small=1,
                       big=10,
                       prior="uniform")
            all_rewards.append(dr)
    finally:
        print("All done!")
        print("---------")
        print("Average discounted reward: %.3f" % (np.mean(all_rewards)))
        print("Std.dev discounted reward: %.3f" % (np.std(all_rewards)))

if __name__ == "__main__":
    main()


