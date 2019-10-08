from maze1d_pomdp import Maze1D_State, Maze1DPOMDP, Maze1D_BeliefState, Maze1D
from pomdp_py import *
import moos3d.util as util
import copy
import math
import time
import random
import sys


def dist(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return math.sqrt((x1-x2)**2 + (y1-y2)**2)

class Maze1D_AbstractBeliefState(Maze1D_BeliefState):
    # This is a belief state regarding a SINGLE target, or a robot.
    def __init__(self, distribution, name="AbstractBeliefState"):
        super().__init__(distribution, name=name)

class Maze1D_AbstractPOMDP(AbstractPOMDP):
    def __init__(self, maze, num_segments, gamma=0.99, allow_search=True):
        if allow_search:
            abstract_actions = ['left', 'right', AbstractPOMDP.SEARCH]
        else:
            abstract_actions = ['left', 'right']
        self.maze = maze
        self._seglen = len(self.maze) // num_segments
        assert self._seglen * num_segments == len(maze), "Segments must have same length. (%d/%d)" % (len(self.maze), num_segments)
        # init_state = self.state_mapper(Maze1D_State(self.maze.robot_pose, self.maze.target_pose))
        self._last_action = None
        super().__init__(abstract_actions, self.transition_func,
                         self.reward_func, self.observation_func,
                         None,
                         # init_state,
                         gamma=gamma)

    def set_prior(self, belief):
        if self.cur_belief is None:
            self.init_belief = belief
            self.cur_belief = belief
            print("Initial belief: %s" % self.init_belief)
        else:
            raise ValueError("This POMDP already has initial belief")

    def action_mapper(self, abstract_action, concrete_state):
        if abstract_action == "left":
            return [-1] * (self.maze.robot_pose % self._seglen + 1)
        elif abstract_action == "right":
            return [1] * (self._seglen - self.maze.robot_pose % self._seglen)
        else: # search
            return []


    def transition_func(self, state, abstract_action):
        """state: Maze1D State"""
        movement = 0
        if abstract_action == "left":
            movement = -1
        elif abstract_action == "right":
            movement = 1
        world_range = (0, len(self.maze)//self._seglen)
        next_robot_pose = self.maze.if_move_by(state.robot_pose, movement, world_range=world_range)
        return Maze1D_State(next_robot_pose, state.target_pose)

    def observation_func(self, next_state, abstract_action):
        # perfect observation model
        if abs(next_state.robot_pose - next_state.target_pose) <= 0:
            return next_state.target_pose
        return -1

    def reward_func(self, state, abstract_action, next_state):
        cur_robot_pose = state.robot_pose
        next_robot_pose = next_state.robot_pose
        reward = 0
        if abstract_action == AbstractPOMDP.SEARCH:
            if next_state.robot_pose == next_state.target_pose: #self.maze.target_pose//self._seglen:
                reward += 10
            else:
                reward -= 10
        else:
            if cur_robot_pose == next_robot_pose:
                reward -= 10
            else:
                reward += 10*math.exp(-abs(next_state.robot_pose - self.maze.target_pose//self._seglen))
        return reward - 0.01        
        # if next_state.robot_pose == self.maze.target_pose:
        #     reward += 10            
        # if cur_robot_pose == next_robot_pose:
        #     reward -= 10
        # return reward - 0.01

    def env_reward_func(self, state, action, next_state):
        reward = 0
        if self.is_in_goal_state():
            reward += 10
        return reward - 0.01
        

    def state_mapper(self, state):
        return Maze1D_State(state.robot_pose//self._seglen, state.target_pose//self._seglen)

    def observation_mapper(self, observations, *params, **kwargs):
        for o in observations:
            if o >= 0:
                return o//self._seglen
        return -1

    def generate_concrete_pomdp(self, abstract_state, gamma=0.8, **kwargs):
        world_range = (self._seglen*(abstract_state.robot_pose),
                       self._seglen*(abstract_state.robot_pose+1))
        pomdp = Maze1DPOMDP(self.maze, gamma=gamma,
                            world_range=world_range,
                            **kwargs)
        return pomdp

    def generate_pomdp_from_abstract_action(self, abstract_action, abstract_state, *params, **kwargs):
        # Because we assume the robot knows its own pose, by taking the abstract
        # action, we know where in the world the robot will end up. So the pomdp
        # will be generated for the area around the robot's new pose
        actions = self.action_mapper(abstract_action, abstract_state)
        if abstract_action == 'left':
            world_range = (self._seglen*(abstract_state.robot_pose - 1),
                           self._seglen*(abstract_state.robot_pose))
        else:
            world_range = (self._seglen*(abstract_state.robot_pose),
                           self._seglen*(abstract_state.robot_pose + 1))
        gamma = kwargs.get("gamma", 0.8)
        pomdp = Maze1DPOMDP(self.maze, gamma=gamma,
                            world_range=world_range,
                            **kwargs)
        print("concrete pomdp world range: %s" % str(world_range))
        return pomdp

    # def execute_agent_action_update_belief(self, abstract_action, abstract_observation=None, **kwargs):
    #     # Abstract POMDP NEVER!! actually execute an action on the Environment; That should be
    #     # done in the Environment!
    #     cur_true_state = self.state_mapper(self.maze.state)
    #     next_true_state = self.transition_func(cur_true_state, abstract_action)
    #     if abstract_observation is None:
    #         abstract_observation = self.observation_func(next_true_state, abstract_action)
    #     reward = self.reward_func(cur_true_state, abstract_action, next_true_state)#env_reward_func(cur_mpe_state, abstract_observation)            
    #     self.belief_update(abstract_action, abstract_observation, **kwargs)
    #     self._last_action = abstract_action
    #     return reward, abstract_observation  # reward and real observation

    def belief_update(self, real_action, real_observation, **kwargs):
        print("updating belief (abstract pomdp)>>>>")
        self.cur_belief.update(real_action, real_observation,
                               self, **kwargs)
        print(self.cur_belief)
        print(">>>>")

    def is_in_goal_state(self):
        return self.maze.robot_pose//self._seglen == self.maze.target_pose//self._seglen\
            and self._last_real_action == AbstractPOMDP.SEARCH
        # mpe = self.cur_belief.distribution.mpe()
        # print(mpe)
        # print("GOAL STATE CHECK %s" % str(mpe.robot_pose == self.maze.target_pose))
        # return mpe.robot_pose == self.maze.target_pose and mpe.target_pose == self.maze.target_pose


    def add_transform(self, state):
        """Used for particle re invigoration"""
        # import pdb; pdb.set_trace()
        state.target_pose = max(0, min(len(self.maze)//self._seglen, state.target_pose + random.randint(-1, 1)))

    def print_true_state(self):
        s = ["."] * (len(self.maze)//self._seglen)
        s[self.maze.robot_pose//self._seglen] = "R"
        s[self.maze.target_pose//self._seglen] = "T"
        print("".join(s))

def pomcp_builder(pomdp, num_particles=1500):
    planner = POMCP(pomdp, num_particles=num_particles,
                    max_time=1.0, max_depth=50, gamma=pomdp.gamma, rollout_policy=_rollout_policy,
                    exploration_const=math.sqrt(2), observation_based_resampling=False)
    return planner

def _rollout_policy(tree, actions):
    return random.choice(actions)

def test_init_belief_correct(abstract_pomdp):
    init_belief = abstract_pomdp.init_belief
    hist = init_belief.distribution.get_histogram()
    # sum to 1
    total_prob = sum(hist[s] for s in hist)
    assert abs(total_prob - 1.0) < 1e-6
    # randomness; no state has too high probability
    max_state = max(hist, key=hist.get)
    max_prob = hist[max_state]
    hist.pop(max_state, None)
    next_max_state = max(hist, key=hist.get)
    assert (max_prob - hist[next_max_state]) < 0.5
    print("OK")

def test_transition_function_correct(abstract_pomdp):
    state = abstract_pomdp.state_mapper(Maze1D_State(0,0))
    next_state = abstract_pomdp.transition_func(state, 'left')
    assert next_state.robot_pose == 0  # didn't move
    state = abstract_pomdp.state_mapper(Maze1D_State(1,0))
    next_state = abstract_pomdp.transition_func(state, 'left')
    assert next_state.robot_pose == 0  # moved only one step
    state = abstract_pomdp.state_mapper(Maze1D_State(2,0))
    next_state = abstract_pomdp.transition_func(state, 'left')
    assert next_state.robot_pose == 0  # moved two steps
    state = abstract_pomdp.state_mapper(Maze1D_State(3,0))
    next_state = abstract_pomdp.transition_func(state, 'left')
    assert next_state.robot_pose == 0  # moved three steps == 1 step    
    state = abstract_pomdp.state_mapper(Maze1D_State(4,0))
    next_state = abstract_pomdp.transition_func(state, 'left')
    assert next_state.robot_pose == 0  # moved three steps == 1 step
    state = abstract_pomdp.state_mapper(Maze1D_State(len(abstract_pomdp.maze)-1,0))
    next_state = abstract_pomdp.transition_func(state, 'right')
    assert next_state.robot_pose == abstract_pomdp.state_mapper(Maze1D_State(len(abstract_pomdp.maze)-1,0)).robot_pose  # didn't move
    print("OK")

# def test_observation_function_correct(abstract_pomdp):
#     maze = Maze1D("R...T")
#     pomdp = Maze1DPOMDP(maze, prior="RANDOM", representation="particles",
#                         num_particles=num_particles, gamma=0.6)
#     init_state = Maze1D_State(maze.robot_pose, maze.robot_pose)
#     init_belief = pomdp.init_belief
#     abstract_pomdp = Maze1D_AbstractPOMDP(maze, init_state, init_belief, gamma=0.6, convert=True)
    
#     abstract_action = "right"
#     pomdp = abstract_pomdp.generate_lower_level_pomdp(abstract_action, num_particles=1000)
    
#     state = pomdp.transition_func(init_state, 1)

class POMDPExperiment:

    def __init__(self, maze, pomdp, planner, max_episodes=100):
        self._maze = maze
        self._planner = planner
        self._abstract_pomdp = pomdp
        self._discounted_sum_rewards = 0
        self._num_iter = 0
        self._max_episodes = max_episodes

    def run(self):
        # self._env.on_loop()
        self._num_iter = 0
        total_time = 0
        rewards = []
        try:
            while not self._abstract_pomdp.is_in_goal_state()\
                  and (self._num_iter < self._max_episodes):

                reward = None

                start_time = time.time()
                abstract_action = self._planner.plan_next_action()
                total_time += time.time() - start_time

                state = self._abstract_pomdp.state_mapper(self._maze.state)
                actions = self._abstract_pomdp.action_mapper(abstract_action, self._maze.state)
                for action in actions:
                    self._maze.state_transition(action)
                next_state = self._abstract_pomdp.state_mapper(self._maze.state)
                abstract_observation, reward = self._abstract_pomdp.real_action_taken(abstract_action, state, next_state)
                # try:
                self._abstract_pomdp.belief_update(abstract_action, abstract_observation, **self._planner.params)
                # except Exception:
                #     import pdb; pdb.set_trace()
                self._planner.update(abstract_action, abstract_observation)                

                # action, reward, observation = \
                #     self._planner.plan_and_execute_next_action()  # the action is a control to the robot
                

                if reward is not None:
                    self._abstract_pomdp.print_true_state()                
                    print("---------------------------------------------")
                    print("%d: Action: %s; Reward: %.3f; Observation: %s"
                          % (self._num_iter, str(abstract_action), reward, str(abstract_observation)))
                    print("---------------------------------------------")
                    self._discounted_sum_rewards += ((self._planner.gamma ** self._num_iter) * reward)
                    rewards.append(reward)

                # self._env._last_observation = self._abstract_pomdp.gridworld.provide_observation()
                self._num_iter += 1
        except KeyboardInterrupt:
            print("Stopped")
            return

        print("Done!")
        return total_time, rewards

    @property
    def discounted_sum_rewards(self):
        return self._discounted_sum_rewards

    @property
    def num_episode(self):
        return self._num_iter
    
    
def unittest(mazestr, num_segments):
    random.seed(100)
    num_particles = 1500
    maze = Maze1D(mazestr)
    pomdp = Maze1DPOMDP(maze, prior="RANDOM", representation="particles",
                        num_particles=num_particles, gamma=0.6)
    pomdp.print_true_state()

    init_state = Maze1D_State(maze.robot_pose, maze.robot_pose)
    init_belief = pomdp.init_belief
    abstract_pomdp = Maze1D_AbstractPOMDP(maze, num_segments, gamma=0.6)
    init_abstract_belief = Maze1D_AbstractBeliefState(init_belief.distribution.__class__(
        init_belief.distribution.get_abstraction(abstract_pomdp.state_mapper)))
    abstract_pomdp.set_prior(init_abstract_belief)

    print(abstract_pomdp.reward_func(Maze1D_State(0,0),'right',Maze1D_State(1,0)))
    print(abstract_pomdp.reward_func(Maze1D_State(0,0),'search',Maze1D_State(0,0)))

    planner = POMCP(abstract_pomdp, num_particles=num_particles,
                    max_time=1.0, max_depth=100, gamma=0.6, rollout_policy=_rollout_policy,
                    exploration_const=math.sqrt(4))  # exploration const helps!!!!!
    experiment = POMDPExperiment(maze, abstract_pomdp, planner, max_episodes=100)
    experiment.run()    

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: %s <mazestr> <num_segments>\n\n"\
              "<mazestr>: Maze string e.g. T....R\n\n"\
              "<num_segments>: number of segments e.g. 5" % (sys.argv[0]))
    else:    
        unittest(sys.argv[1], int(sys.argv[2]))
