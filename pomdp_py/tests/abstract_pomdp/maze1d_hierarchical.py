from maze1d_pomdp import Maze1D_State, Maze1DPOMDP, Maze1D_BeliefState, Maze1D
from maze1d_abstract_pomdp import Maze1D_AbstractPOMDP, Maze1D_AbstractBeliefState
from pomdp_py import *
import copy
import random
import sys
import math


# class AbstractPOMDPPlanner(Planner):
#     def __init__(self, maze, abstract_pomdp, num_particles=1000):
#         self._abstract_pomdp = abstract_pomdp
#         self._abstract_pomdp_planner = POMCP(abstract_pomdp, num_particles=num_particles,
#                                        max_time=1.0, max_depth=100, gamma=0.6, rollout_policy=_rollout_policy,
#                                        exploration_const=math.sqrt(4))  # exploration const helps!!!!!
#         self._maze = maze
#         self._num_particles = num_particles
#         self._plan_level = "abstract"
#         self._pending_actions = []
#         self._accumulating_observations = []
#         self._concrete_pomdp = None
#         self._concrete_pomdp_planner = None
#         self.cur_abstract_action = None

#     def plan_next_action(self):
#         if self._plan_level == "abstract":
#             if len(self._pending_actions) == 0:
#                 abstract_action = self._abstract_pomdp_planner.plan_next_action()
#                 self.cur_abstract_action = abstract_action
#                 if abstract_action != AbstractPOMDP.SEARCH:
#                     self._pending_actions = self._abstract_pomdp.action_mapper(abstract_action, self._maze.state)
#                     action = self._pending_actions.pop(0)
#                 else:
#                     self._plan_level = "concrete"
#                     self._concrete_pomdp = self._abstract_pomdp.generate_concrete_pomdp(self._abstract_pomdp.state_mapper(self._maze.state),
#                                                                                         num_particles=self._num_particles)  # techinically, concrete pomdp can have different num_particles
#             else:
#                 action = self._pending_actions.pop(0)
#                 if len(self._pending_actions) == 0:
#                     self._plan_level = "abstract"
                    
#         if self._plan_level == "concrete":
#             if self._concrete_pomdp_planner is None:
#                 self._concrete_pomdp_planner = POMCP(self._concrete_pomdp_planner, num_particles=self._num_particles,
#                                                      max_time=1.0, max_depth=100, gamma=0.6, rollout_policy=_rollout_policy,
#                                                      exploration_const=math.sqrt(4))  # exploration const helps!!!!!
#             action = self._concrete_pomdp_planner.plan_next_action()
#         return action

#     def update(self, real_action, real_observation):
#         if self._plan_level == "abstract":
#             self._abstract_pomdp_planner.update(real_action, real_observation)
#             if real_action == AbstractPOMDP.SEARCH:
#                 self._plan_level = "concrete"
#         else:
#             self._concrete_pomdp_planner.update(real_action, real_observation)
#             if real_action == AbstractPOMDP.BACKTRACK:
#                 self._plan_level = "abstract"


# class POMDPExperiment:

#     def __init__(self, maze, abstract_pomdp, fullpomdp, abstract_planner, max_episodes=100):
#         self._maze = maze
#         self._abstract_planner = abstract_planner
#         self._abstract_pomdp = abstract_pomdp
#         self._pomdp = fullpomdp
#         self._discounted_sum_rewards = 0
#         self._num_iter = 0
#         self._max_episodes = max_episodes

#     def run(self):
#         # self._env.on_loop()
#         self._num_iter = 0
#         total_time = 0
#         rewards = []

#         accumulating_observations = []
#         try:
#             while not self._abstract_pomdp.is_in_goal_state()\
#                   and (self._num_iter < self._max_episodes):

#                 reward = None

#                 start_time = time.time()
#                 action = self._planner.plan_next_action()
#                 total_time += time.time() - start_time

#                 # execute the action, and transition state.
#                 state = copy.deepcopy(self._maze.state)
#                 self._maze.state_transition(action)
#                 next_state = copy.deepcopy(self._maze.state)

#                 # accumulate the observation
#                 observation, reward = self._pomdp.real_action_taken(action, state, next_state)
#                 self.accumulating_observations.append(observation)

#                 if action == AbstractPOMDP.BACKTRACK:
#                     abstract_observation = self._abstract_pomdp.observation_mapper(self._accumulating_observations)
#                     self._abstract_pomdp.belief_update(self._planner.cur_abstract_action,
#                                                        abstract_observation)
#                     sim_abstract_observation, reward = self._abstract_pomdp.real_action_taken(abstract_action, state, next_state)
#                     assert sim_abstract_observation == abstract_observation, "sim!=real; belief update will suffer"
#                     self._planner.update(self._planner.cur_abstract_action, abstract_observation)
#                 else:
#                     self._planner.update(action, observation)

#                 if reward is not None:
#                     self._abstract_pomdp.print_true_state()                
#                     print("---------------------------------------------")
#                     print("%d: Action: %s; Reward: %.3f; Cumulative Reward: %.3f;  Observation: %s"
#                           % (self._num_iter, str(action), reward, self._discounted_sum_rewards, str(observation)))
#                     print("---------------------------------------------")
#                     self._discounted_sum_rewards += reward
#                     rewards.append(reward)

#                 # self._env._last_observation = self._abstract_pomdp.gridworld.provide_observation()
#                 self._num_iter += 1
#         except KeyboardInterrupt:
#             print("Stopped")
#             return

#         print("Done!")
#         return total_time, rewards

#     @property
#     def discounted_sum_rewards(self):
#         return self._discounted_sum_rewards

#     @property
#     def num_episode(self):
#         return self._num_iter

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
    while not (maze.robot_pose == maze.target_pose  # overall goal
               or num_iter >= max_iter):
        print("=Abstract world===")
        print_true_state(maze, seglen=abstract_pomdp._seglen)
        print("=True world=======")
        print_true_state(maze, seglen=1)
        
        abstract_state = abstract_pomdp.state_mapper(maze.state)
        abstract_action = planner.plan_next_action()

        # execute abstract action
        actions = abstract_pomdp.action_mapper(abstract_action, maze.state)
        for action in actions:
            maze.state_transition(action)
        next_abstract_state = abstract_pomdp.state_mapper(maze.state)

        # after executing abstract action, plan with concrete pomdp in resulting region
        num_particles = 1000
        concrete_pomdp = abstract_pomdp.generate_concrete_pomdp(abstract_pomdp.state_mapper(maze.state),
                                                                      num_particles=num_particles)
        concrete_planner = POMCP(concrete_pomdp, num_particles=num_particles,
                                 max_time=1.0, max_depth=100, gamma=0.6, rollout_policy=_rollout_policy,
                                 exploration_const=math.sqrt(4))  # exploration const helps!!!!!
        action = concrete_planner.plan_next_action()
        state = copy.deepcopy(maze.state)
        next_state = maze.state_transition(action)
        observation, reward = concrete_pomdp.real_action_taken(action, state, next_state)
        concrete_pomdp.belief_update(action, observation, **planner.params)
        concrete_planner.update(action, observation)
        observations = [observation]
        print_info(action, observation, reward)

        while not (concrete_pomdp.is_in_goal_state()
                   or action == AbstractPOMDP.BACKTRACK):
            print_true_state(maze, seglen=1)
            action = concrete_planner.plan_next_action()
            state = copy.deepcopy(maze.state)
            next_state = maze.state_transition(action)
            observation, reward = concrete_pomdp.real_action_taken(action, state, next_state)
            concrete_pomdp.belief_update(action, observation, **concrete_planner.params)
            concrete_planner.update(action, observation)
            observations.append(observation)
            print_info(action, observation, reward)

        if action == AbstractPOMDP.BACKTRACK:
            print("BACKTRACK!")
            # udpate abstract pomdp belief using accumulated obseravtions
            abstract_observation = abstract_pomdp.observation_mapper(observations)
            abstract_pomdp.belief_update(abstract_action, abstract_observation, **planner.params)
            planner.update(abstract_action, abstract_observation)
        print_info(abstract_action, abstract_observation, reward)
        num_iter += 1
        

def plan_abstract_SEARCH(maze, abstract_pomdp, planner, max_iter=50):
    num_iter = 0
    while not (maze.robot_pose == maze.target_pose  # overall goal
               or num_iter >= max_iter):
        print("=Abstract world===")
        print_true_state(maze, seglen=abstract_pomdp._seglen)
        print("=True world=======")
        print_true_state(maze, seglen=1)
        
        abstract_state = abstract_pomdp.state_mapper(maze.state)
        abstract_action = planner.plan_next_action()

        if abstract_action == AbstractPOMDP.SEARCH:
            print("SEARCH!")
            
            num_particles = 1000
            concrete_pomdp = abstract_pomdp.generate_concrete_pomdp(abstract_pomdp.state_mapper(maze.state),
                                                                          num_particles=num_particles)
            concrete_planner = POMCP(concrete_pomdp, num_particles=num_particles,
                                     max_time=1.0, max_depth=100, gamma=0.6, rollout_policy=_rollout_policy,
                                     exploration_const=math.sqrt(4))  # exploration const helps!!!!!
            action = concrete_planner.plan_next_action()
            state = copy.deepcopy(maze.state)
            next_state = maze.state_transition(action)
            observation, reward = concrete_pomdp.real_action_taken(action, state, next_state)
            concrete_pomdp.belief_update(action, observation, **planner.params)
            concrete_planner.update(action, observation)
            observations = [observation]
            print_info(action, observation, reward)

            while not (concrete_pomdp.is_in_goal_state()
                       or action == AbstractPOMDP.BACKTRACK):
                print_true_state(maze, seglen=1)
                action = concrete_planner.plan_next_action()
                state = copy.deepcopy(maze.state)
                next_state = maze.state_transition(action)
                observation, reward = concrete_pomdp.real_action_taken(action, state, next_state)
                concrete_pomdp.belief_update(action, observation, **planner.params)
                concrete_planner.update(action, observation)
                observations.append(observation)
                print_info(action, observation, reward)

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
            next_abstract_state = abstract_pomdp.state_mapper(maze.state)
            # TODO: this 'abstract_obseravtion' is given mysteriously - no world can directly provide abstract observations.
            abstract_observation, reward = abstract_pomdp.real_action_taken(abstract_action, abstract_state, next_abstract_state)
            abstract_pomdp.belief_update(abstract_action, abstract_observation, **planner.params)
            planner.update(abstract_action, abstract_observation)
        print_info(abstract_action, abstract_observation, reward)
        num_iter += 1

def _rollout_policy(tree, actions):
    return random.choice(actions)

def unittest():
    random.seed(100)
    num_particles = 1500
    maze = Maze1D(sys.argv[1])
    pomdp = Maze1DPOMDP(maze, prior="RANDOM", representation="particles",
                        num_particles=num_particles, gamma=0.6)
    pomdp.print_true_state()
    
    num_segments = int(sys.argv[2])

    allow_search = True
    if len(sys.argv) > 3:
        if sys.argv[3] == "--no-search":
            allow_search = False

    init_state = Maze1D_State(maze.robot_pose, maze.robot_pose)
    init_belief = pomdp.init_belief
    abstract_pomdp = Maze1D_AbstractPOMDP(maze, num_segments, gamma=0.6, allow_search=allow_search)
    init_abstract_belief = Maze1D_AbstractBeliefState(init_belief.distribution.__class__(
        init_belief.distribution.get_abstraction(abstract_pomdp.state_mapper)))
    abstract_pomdp.set_prior(init_abstract_belief)

    print(abstract_pomdp.reward_func(Maze1D_State(0,0),'right',Maze1D_State(1,0)))
    print(abstract_pomdp.reward_func(Maze1D_State(0,0),'search',Maze1D_State(0,0)))

    planner = POMCP(abstract_pomdp, num_particles=num_particles,
                    max_time=1.0, max_depth=100, gamma=0.6, rollout_policy=_rollout_policy,
                    exploration_const=math.sqrt(4))  # exploration const helps!!!!!

    if allow_search:
        plan_abstract_SEARCH(maze, abstract_pomdp, planner, max_iter=50)
    else:
        plan_abstract_BACKTRACK(maze, abstract_pomdp, planner, max_iter=50)

if __name__ == '__main__':
    unittest()
