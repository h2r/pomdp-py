from maze1d_pomdp import Maze1D_State, Maze1DPOMDP, Maze1D_BeliefState, Maze1D
from maze1d_abstract_pomdp import Maze1D_AbstractPOMDP, Maze1D_AbstractBeliefState
from pomdp_py import *
import copy
import random
import sys
import math
import time

POMCP_PLANNING_TIME = 0.5

def print_true_state(maze, seglen=1):
    s = ["."] * (len(maze)//seglen)
    s[maze.robot_pose//seglen] = "R"
    s[maze.target_pose//seglen] = "T"
    print("".join(s))

def print_info(action, observation, reward):
    print("---------------------------------------------")
    print("Action: %s; Reward: %.3f; Observation: %s"
          % (str(action), reward, str(observation)))
    print("---------------------------------------------")

def plan_abstract_BACKTRACK(maze, abstract_pomdp, planner, max_iter=50):
    num_iter = 0
    total_time = 0
    rewards = []
    while not (maze.robot_pose == maze.target_pose  # overall goal
               or num_iter >= max_iter):
        print("=Abstract world===")
        print_true_state(maze, seglen=abstract_pomdp._seglen)
        print("=True world=======")
        print_true_state(maze, seglen=1)
        
        abstract_state = abstract_pomdp.state_mapper(maze.state)
        start_time = time.time()
        abstract_action = planner.plan_next_action()
        total_time += time.time() - start_time        

        # execute abstract action
        actions = abstract_pomdp.action_mapper(abstract_action, maze.state)
        for action in actions:
            maze.state_transition(action)
            rewards.append(-0.01)  # step cost
        next_abstract_state = abstract_pomdp.state_mapper(maze.state)

        # after executing abstract action, plan with concrete pomdp in resulting region
        num_particles = 1000
        concrete_pomdp = abstract_pomdp.generate_concrete_pomdp(abstract_pomdp.state_mapper(maze.state),
                                                                      num_particles=num_particles)
        concrete_planner = POMCP(concrete_pomdp, num_particles=num_particles,
                                 max_time=POMCP_PLANNING_TIME, max_depth=100, gamma=0.6, rollout_policy=_rollout_policy,
                                 exploration_const=math.sqrt(4))  # exploration const helps!!!!!
        start_time = time.time()        
        action = concrete_planner.plan_next_action()
        total_time += time.time() - start_time
        
        state = copy.deepcopy(maze.state)
        next_state = maze.state_transition(action)
        observation, reward = concrete_pomdp.real_action_taken(action, state, next_state)
        concrete_pomdp.belief_update(action, observation, **planner.params)
        concrete_planner.update(action, observation)
        observations = [observation]
        print_info(action, observation, reward)
        rewards.append(reward)
        num_iter += 1

        while not (concrete_pomdp.is_in_goal_state()
                   or action == AbstractPOMDP.BACKTRACK):
            print_true_state(maze, seglen=1)
            
            start_time = time.time()        
            action = concrete_planner.plan_next_action()
            total_time += time.time() - start_time
            
            state = copy.deepcopy(maze.state)
            next_state = maze.state_transition(action)
            observation, reward = concrete_pomdp.real_action_taken(action, state, next_state)
            concrete_pomdp.belief_update(action, observation, **concrete_planner.params)
            concrete_planner.update(action, observation)
            observations.append(observation)
            print_info(action, observation, reward)
            rewards.append(reward)
            num_iter += 1            

        abstract_observation = None            
        if action == AbstractPOMDP.BACKTRACK:
            print("BACKTRACK!")
            # udpate abstract pomdp belief using accumulated obseravtions
            abstract_observation = abstract_pomdp.observation_mapper(observations)
            abstract_pomdp.belief_update(abstract_action, abstract_observation, **planner.params)
            planner.update(abstract_action, abstract_observation)
        print_info(abstract_action, abstract_observation, reward)
    return total_time, rewards
        

def plan_abstract_SEARCH(maze, abstract_pomdp, planner, max_iter=50):
    num_iter = 0
    total_time = 0
    rewards = []
    while not (maze.robot_pose == maze.target_pose  # overall goal
               or num_iter >= max_iter):
        print("=Abstract world===")
        print_true_state(maze, seglen=abstract_pomdp._seglen)
        print("=True world=======")
        print_true_state(maze, seglen=1)

        abstract_observation = None
        abstract_state = abstract_pomdp.state_mapper(maze.state)
        start_time = time.time()
        abstract_action = planner.plan_next_action()
        total_time += time.time() - start_time

        if abstract_action == AbstractPOMDP.SEARCH:
            print("SEARCH!")
            
            num_particles = 1000
            concrete_pomdp = abstract_pomdp.generate_concrete_pomdp(abstract_pomdp.state_mapper(maze.state),
                                                                          num_particles=num_particles)
            concrete_planner = POMCP(concrete_pomdp, num_particles=num_particles,
                                     max_time=POMCP_PLANNING_TIME, max_depth=100, gamma=0.6, rollout_policy=_rollout_policy,
                                     exploration_const=math.sqrt(4))  # exploration const helps!!!!!
            start_time = time.time()
            action = concrete_planner.plan_next_action()
            total_time += time.time() - start_time
            
            state = copy.deepcopy(maze.state)
            next_state = maze.state_transition(action)
            observation, reward = concrete_pomdp.real_action_taken(action, state, next_state)
            concrete_pomdp.belief_update(action, observation, **planner.params)
            concrete_planner.update(action, observation)
            observations = [observation]
            print_info(action, observation, reward)
            rewards.append(reward)
            num_iter += 1

            while not (concrete_pomdp.is_in_goal_state()
                       or action == AbstractPOMDP.BACKTRACK):
                print_true_state(maze, seglen=1)
                
                start_time = time.time()
                action = concrete_planner.plan_next_action()
                total_time += time.time() - start_time
                
                state = copy.deepcopy(maze.state)
                next_state = maze.state_transition(action)
                observation, reward = concrete_pomdp.real_action_taken(action, state, next_state)
                concrete_pomdp.belief_update(action, observation, **planner.params)
                concrete_planner.update(action, observation)
                observations.append(observation)
                print_info(action, observation, reward)
                rewards.append(reward)
                num_iter += 1                

            if action == AbstractPOMDP.BACKTRACK:
                print("BACKTRACK!")
                next_abstract_state = abstract_pomdp.state_mapper(maze.state)
                abstract_observation = abstract_pomdp.observation_mapper(observaitons)
                abstract_pomdp.belief_update(abstract_action, abstract_observation)
                planner.update(abstract_action, abstract_observation)
        else:
            actions = abstract_pomdp.action_mapper(abstract_action, maze.state)
            for action in actions:
                maze.state_transition(action)
                rewards.append(-0.01)  # step cost
            next_abstract_state = abstract_pomdp.state_mapper(maze.state)
            # TODO: this 'abstract_obseravtion' is given mysteriously - no world can directly provide abstract observations.
            abstract_observation, reward = abstract_pomdp.real_action_taken(abstract_action, abstract_state, next_abstract_state)
            abstract_pomdp.belief_update(abstract_action, abstract_observation, **planner.params)
            planner.update(abstract_action, abstract_observation)
            rewards.append(reward)
        print_info(abstract_action, abstract_observation, reward)
    return total_time, rewards

def _rollout_policy(tree, actions):
    return random.choice(actions)

def unittest(mazestr, num_segments, allow_search=True):
    random.seed(100)
    num_particles = 1000
    maze = Maze1D(mazestr)
    pomdp = Maze1DPOMDP(maze, prior="RANDOM", representation="particles",
                        num_particles=num_particles, gamma=0.6)
    pomdp.print_true_state()
    init_state = Maze1D_State(maze.robot_pose, maze.robot_pose)
    init_belief = pomdp.init_belief
    abstract_pomdp = Maze1D_AbstractPOMDP(maze, num_segments, gamma=0.6, allow_search=allow_search)
    init_abstract_belief = Maze1D_AbstractBeliefState(init_belief.distribution.__class__(
        init_belief.distribution.get_abstraction(abstract_pomdp.state_mapper)))
    abstract_pomdp.set_prior(init_abstract_belief)

    print(abstract_pomdp.reward_func(Maze1D_State(0,0),'right',Maze1D_State(1,0)))
    print(abstract_pomdp.reward_func(Maze1D_State(0,0),'search',Maze1D_State(0,0)))

    planner = POMCP(abstract_pomdp, num_particles=num_particles,
                    max_time=POMCP_PLANNING_TIME, max_depth=100, gamma=0.6, rollout_policy=_rollout_policy,
                    exploration_const=math.sqrt(4))  # exploration const helps!!!!!

    if allow_search:
        total_time, rewards = plan_abstract_SEARCH(maze, abstract_pomdp, planner, max_iter=100)
    else:
        total_time, rewards = plan_abstract_BACKTRACK(maze, abstract_pomdp, planner, max_iter=100)
    print("Total planning time: %.3fs; Rewards: %.3f" % (total_time, sum(rewards)))
    return total_time, rewards

if __name__ == '__main__':
    allow_search = True
    if len(sys.argv) > 3:
        if sys.argv[3] == "--no-search":
            allow_search = False
    unittest(sys.argv[1], int(sys.argv[2]), allow_search)
