# Test POMCP on an MDP
from pomdp_py import *
from maze2d import GridWorld, Environment, dist
import pygame
import cv2
import math
import numpy as np
import random
import sys
import time
from collections import defaultdict

# Simple gridworld MDP. The goal is to reach target location in a 2d gridworld.

class Maze2D_State(State):
    def __init__(self, robot_pose, target_pose):
        self.robot_pose = robot_pose
        self.target_pose = target_pose
        super().__init__(data=[robot_pose, target_pose])
    def unpack(self):
        return self.robot_pose, self.target_pose
    def __str__(self):
        return 'Maze2D_State::{}'.format(self.data)
    def __repr__(self):
        return self.__str__()


class Maze2D_BeliefDistribution(BeliefDistribution):
    def __init__(self, maze2dpomdp):
        self._maze2dpomdp = maze2dpomdp
        self._distribution_target_pose = defaultdict(lambda: 1)

    def __getitem__(self, state):
        """We assume that the agent knows its own pose. Therefore, if the robot_pose
        in `state` is not equal to the gridworld's robot pose, then return 0. Otherwise,
        return a constant (we're not taking care of the normalization here). If the state
        has incorrect 'object_detected' then return 0 as well."""
        if state.robot_pose != self._maze2dpomdp.gridworld.robot_pose:
            return 0
        else:
            return self._distribution_target_pose[state.target_pose] # just a constant

    def __setitem__(self, state, value):
        _, target_pose = state.unpack()
        self._distribution_target_pose[target_pose] = value

    def __len__(self):
        """This could be a huge number"""
        gridworld = self._maze2dpomdp.gridworld
        w, h = gridworld.width, gridworld.height
        dof_orientation = 7 # 45-degree increment
        return ((w*h)*dof_orientation**3) * (w*h)

    def __str__(self):
        return "MOS3D_BeliefDistribution(robot_pose:%s)" % str(self._maze2dpomdp.gridworld.robot_pose)

    def __hash__(self):
        keys = tuple(self._distribution_target_pose.keys())
        return hash(keys)

    def __eq__(self):
        if isinstance(other, BeliefDistribution):
            return self._distribution_target_pose == other._distribution_target_pose
        return False

    def mpe(self):
        """Most Probable Explanation; i.e. the state with the highest probability"""
        robot_pose = self._maze2dpomdp.gridworld.robot_pose
        target_pose = None
        if len(self._distribution_target_pose) > 0:
            pose = max(self._distribution_target_pose,
                       key=self._distribution_target_pose.get,
                       default=random.choice(list(self._distribution_target_pose)))
            if self._distribution_target_pose[target_pose] > 1:
                target_pose = pose

        if target_pose is None:
            # No state has been associated with a probability. So they all have
            # the same. Thus generate a random state for the objects.
            # target_pose = GridWorld.target_pose_to_tuple({
            #     objid:(random.randint(0, self.gridworld.width-1),
            #            random.randint(0, self.gridworld.length-1),
            #            random.randint(0, self.gridworld.height-1))
            #     for objid in self.gridworld.objects
            # })
            target_pose = self.gridworld.target_pose
        return Maze2D_State(robot_pose, target_pose)

    def random(self):
        robot_pose = self._maze2dpomdp.gridworld.robot_pose
        target_pose = self.gridworld.target_pose
        return Maze2D_State(robot_pose, target_pose)
    
    @property
    def gridworld(self):
        return self._maze2dpomdp.gridworld
    

class Maze2D_BeliefState(BeliefState):
    def __init__(self, maze2dpomdp, distribution=None):
        if distribution is None:
            super().__init__(Maze2D_BeliefDistribution(maze2dpomdp))
        else:
            super().__init__(distribution)

    @property
    def gridworld(self):
        return self.distribution.gridworld

    def sample(self, sampling_method='random'):
        if sampling_method == 'random':  # random uniform
            robot_pose = self.gridworld.robot_pose
            # target_pose = {
            #     objid:(random.randint(0, self.gridworld.width-1),
            #            random.randint(0, self.gridworld.length-1),
            #            random.randint(0, self.gridworld.height-1))
            #     for objid in self.gridworld.objects
            # }
            target_pose = self.gridworld.target_pose
            return Maze2D_State(robot_pose, target_pose)
        elif sampling_method == 'max':
            return self.distribution.mpe()
        raise NotImplementedError('Sampling method {} not implemented yet'.format(sampling_method))            
        

class Maze2D_MDP(POMDP):
    """
    Even though this is a child class of POMDP, it is in fact
    an MDP, with an observation function that does not affect
    transition or reward. This is merely used to verify if
    the POMCP algorithm works properly - if properly, it should
    be able to solve this MDP as if it is just navigation path
    planning.
    """
    ACTIONS = {
        0:(1, 0),  # forward
        1:(-1, 0), # backward
        2:(0, -math.pi/4),  # left 45 degree
        3:(0, math.pi/4)    # right 45 degree
    }

    def __init__(self, gridworld, sensor_params):
        self._gridworld = gridworld
        self._sensor_params = sensor_params
        init_true_state = Maze2D_State(self._gridworld.robot_pose, self._gridworld.target_pose)
        b0 = Maze2D_BeliefState(self)
        super().__init__(list(Maze2D_MDP.ACTIONS.keys()),
                         self._transition_func,
                         self._reward_func,
                         self._observation_func,
                         b0,
                         init_true_state)

    @property
    def gridworld(self):
        return self._gridworld

    def _transition_func(self, state, action):
        state_robot = state.robot_pose
        action = Maze2D_MDP.ACTIONS[action]
        next_state_robot = self.gridworld.if_move_by(state_robot, action[0], action[1])
        next_state_target = state.target_pose
        return Maze2D_State(next_state_robot, next_state_target)

    def _observation_func(self, next_state, action):
        next_state_robot = next_state.robot_pose
        observation = tuple(map(tuple, self.gridworld.if_observe_at(next_state_robot, self._sensor_params, known_correspondence=True)))
        return observation
    
    def _reward_func(self, state, action, next_state):
        next_state_robot = next_state.robot_pose
        rx, ry, rth = next_state_robot
        # reward = 0
        # if self.gridworld.target_pose == (rx, ry):
        #     reward += 1
        # reward = math.exp(-1.5*dist((rx,ry), self.gridworld.target_pose))
        reward = 0
        if (rx, ry) == self.gridworld.target_pose:
            reward += 1
        return reward - 0.05

    def execute_agent_action_update_belief(self, action, **kwargs):
        """Completely overriding parent's function. There is no belief update here."""
        # TODO: cur_state IS WRONG! NOT a POMDP        
        self.cur_state = Maze2D_State(self._gridworld.robot_pose, self._gridworld.target_pose)
        next_state = self.transition_func(self.cur_state, action)
        observation = self.observation_func(next_state, action)
        reward = self.reward_func(self.cur_state, action, next_state)
        self._gridworld.move_robot(*Maze2D_MDP.ACTIONS[action])
        self.cur_state = next_state
        return reward, observation

    def is_in_goal_state(self):
        return self.cur_state.robot_pose[:2] == self.gridworld.target_pose

    def update_belief(self, real_action, real_observation, **kwargs):
        pass


class Experiment:

    def __init__(self, env, pomdp, planner, render=True, max_episodes=100):
        self._env = env
        self._planner = planner
        self._pomdp = pomdp
        self._discounted_sum_rewards = 0
        self._num_iter = 0
        self._max_episodes = max_episodes

    def run(self):
        if self._env.on_init() == False:
            raise Exception("Environment failed to initialize")

        # self._env.on_loop()
        self._num_iter = 0
        self._env.on_render()

        total_time = 0
        rewards = []
        try:
            while self._env._running \
                  and not self._pomdp.is_in_goal_state()\
                  and self._num_iter < self._max_episodes:
                # for event in pygame.event.get():
                #     self._env.on_event(event)
                start_time = time.time()
                action, reward, observation = \
                    self._planner.plan_and_execute_next_action()  # the action is a control to the robot
                total_time += time.time() - start_time
                print("%d: Action: %s; Reward: %.3f; Observation: %s"
                      % (self._num_iter, str(action), reward, str(observation)))
                self._discounted_sum_rewards += ((self._planner.gamma ** self._num_iter) * reward)
                rewards.append(reward)

                # self._env._last_observation = self._pomdp.gridworld.provide_observation()
                self._env.on_loop()
                self._env.on_render()
                self._num_iter += 1
        except KeyboardInterrupt:
            print("Stopped")
            self._env.on_cleanup()
            return

        print("Done!")
        time.sleep(1)
        self._env.on_cleanup()
        return total_time, rewards

    @property
    def discounted_sum_rewards(self):
        return self._discounted_sum_rewards

    @property
    def num_episode(self):
        return self._num_iter

world0 = \
"""
R..
...
..T
"""

world1 = \
"""
Rx...
.x.xT
.....
"""

world2= \
"""
............................
..xxxxxxxxxxxxxxxxxxxxxxxx..
..xR.....................x..
..x......................x..
..x..xxxxxxxxxxxxxxxxxx..x..
..x..x................x..x..
..x..xxxxxxxxxxxxxxxxxx..x..
..x......................x..
..x....................T.x..
..xxxxxxxxxxxxxxxxxxxxxxxx..
............................
"""

world3 = \
"""
....................................
....................................
............R.......................
....................................
....................................
.....x.......x.......x.......x......
....................................
....................................
....................................
....................................
....................................
....................................
....................................
.....x.......x.......x.......x......
....................................
....................................
.............................T......
....................................
....................................
""" # textbook

world4 = \
"""
.T........R...
"""


world5 = \
"""
RT
"""


def _rollout_policy(tree, actions):
    return random.choice(actions)

def unittest():
    gridworld = GridWorld(world1)
    sys.stdout.write("Initializing POMDP...")
    env = Environment(gridworld,
                      sensor_params={
                          'max_range':5,
                          'min_range':1,
                          'view_angles': math.pi/2,
                          'sigma_dist': 0.01,
                          'sigma_bearing': 0.01},
                      res=30, fps=10, controllable=False)
    # env.on_execute()
    sys.stdout.write("done\n")
    sys.stdout.write("Initializing Planner: POMCP...")
    pomdp = Maze2D_MDP(gridworld, env.sensor_params)
    sys.stdout.write("Initializing Planner: POMCP...")    
    planner = POMCP(pomdp, max_time=1.5, max_depth=50, gamma=0.99, rollout_policy=_rollout_policy,
                    exploration_const=math.sqrt(2))
    sys.stdout.write("done\n")    
    experiment = Experiment(env, pomdp, planner)
    print("Running experiment")
    experiment.run()

if __name__ == '__main__':
    unittest()
