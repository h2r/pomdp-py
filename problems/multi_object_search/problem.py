# Uses the domain, models, and agent/environment
# to actually define the POMDP problem for multi-object search.
# Then, solve it using POUCT or POMCP.
import pomdp_py
from env.env import *
from env.visual import *
from agent.agent import *
from example_worlds import *
from domain.observation import *
import argparse
import time
import random

class MosOOPOMDP(pomdp_py.OOPOMDP):
    """
    A MosOOPOMDP is instantiated given a string description
    of the search world, sensor descriptions for robots,
    and the necessary parameters for the agent's models.

    Note: This is of course a simulation, where you can
    generate a world and know where the target objects are
    and then construct the Environment object. But in the
    real robot scenario, you don't know where the objects
    are. In that case, as I have done it in the past, you
    could construct an Environment object and give None to
    the object poses.
    """
    def __init__(self, robot_id, env=None, grid_map=None,
                 sensors=None, sigma=0, epsilon=1,
                 belief_rep="histogram", prior={}, num_particles=100):
        """
        Args:
            robot_id (int or str): the id of the agent that will solve this MosOOPOMDP.
                If it is a `str`, it will be interpreted as an integer using `interpret_robot_id`
                in env/env.py.
            env (MosEnvironment): the environment. 
            grid_map (str): Search space description. See env/env.py:interpret. An example:
                rx...
                .x.xT
                .....
                Ignored if env is not None
            sensors (dict): map from robot character to sensor string.
                For example: {'r': 'laser fov=90 min_range=1 max_range=5
                                    angle_increment=5'}
                Ignored if env is not None

            sigma, epsilon: observation model paramters
            belief_rep (str): belief representation. Either histogram or particles.
            prior (dict or str): either a dictionary as defined in agent/belief.py
                or a string, either "uniform" or "informed". For "uniform", a uniform
                prior will be given. For "informed", a perfect prior will be given.
            num_particles (int): setting for the particle belief representation
        """
        if env is None:
            assert grid_map is not None and sensors is not None,\
                "Since env is not provided, you must provide string descriptions"\
                "of the world and sensors."
            worldstr = equip_sensors(grid_map, sensors)
            env = interpret(worldstr)

        # construct prior
        if type(prior) == str:
            if prior == "uniform":
                prior = {}
            elif prior == "informed":
                for objid in env.target_states:
                    groundtruth_pose = env.state.pose(objid)
                    prior[groundtruth_pose] = 1.0

        # Potential extension: a multi-agent POMDP. For now, the environment
        # can keep track of the states of multiple agents, but a POMDP is still
        # only defined over a single agent. Perhaps, MultiAgent is just a kind
        # of Agent, which will make the implementation of multi-agent POMDP cleaner.
        robot_id = robot_id if type(robot_id) == int else interpret_robot_id(robot_id)
        agent = MosAgent(robot_id,
                         env.state.object_states[robot_id],
                         env.target_objects,
                         (env.width, env.length),
                         env.sensors[robot_id],
                         sigma=sigma,
                         epsilon=epsilon,
                         belief_rep=belief_rep,
                         prior=prior,
                         num_particles=num_particles)
        super().__init__(agent, env,
                         name="MOS(%d,%d,%d)" % (env.width, env.length, len(env.target_objects)))

### Solve the problem with POUCT/POMCP planner ###
### This is the main online POMDP solver logic ###
def solve(problem,
          max_depth=10,  # planning horizon
          discount_factor=0.99,
          planning_time=1.,       # amount of time (s) to plan each step
          exploration_const=1000, # exploration constant
          visualize=True,
          max_time=120,  # maximum amount of time allowed to solve the problem
          max_steps=500): # maximum number of planning steps the agent can take.
    """
    This function terminates when:
    - maximum time (max_time) reached; This time includes planning and updates
    - agent has planned `max_steps` number of steps
    - agent has taken n FindAction(s) where n = number of target objects.

    Args:
        visualize (bool) if True, show the pygame visualization.
    """

    random_objid = random.sample(problem.env.target_objects, 1)[0]
    random_object_belief = problem.agent.belief.object_belief[random_objid]
    if isinstance(random_object_belief, pomdp_py.Histogram):
        # Use POUCT
        belief_rep = "histogram"
        planner = pomdp_py.POUCT(max_depth=max_depth,
                                 discount_factor=discount_factor,
                                 planning_time=planning_time,
                                 exploration_const=exploration_const,
                                 rollout_policy=problem.agent.policy_model)  # Random by default
    elif isinstance(random_object_belief, pomdp_py.Particles):
        # Use POMCP
        belief_rep = "particles"
        planner = pomdp_py.POMCP(max_depth=max_depth,
                                 discount_factor=discount_factor,
                                 planning_time=planning_time,
                                 exploration_const=exploration_const,
                                 rollout_policy=problem.agent.policy_model)  # Random by default
    else:
        raise ValueError("Unsupported object belief type %s" % str(type(random_object_belief)))
    
    if visualize:
        viz = MosViz(env, controllable=False)  # controllable=False means no keyboard control.
        if viz.on_init() == False:
            raise Exception("Environment failed to initialize")
        viz.on_render()

    _time_used = 0
    _find_actions_count = 0
    _total_reward = 0  # total, undiscounted reward
    for i in range(max_steps):
        # Plan action
        _start = time.time()
        real_action = planner.plan(problem.agent)
        _time_used += time.time() - _start
        if _time_used > max_time:
            break  # no more time to update.

        # Execute action
        reward = problem.env.state_transition(real_action, execute=True)

        # Receive observation
        _start = time.time()
        real_observation = env.provide_observation(problem.agent.observation_model, real_action)

        # Updates
        agent.clear_history()  # truncate history
        agent.update_history(real_action, real_observation)
        # For particles belief, belief update happens together with planner (POMCP) update;
        if belief_rep == "histogram":
            new_belief = pomdp_py.update_histogram_belief(problem.agent.cur_belief,
                                                          real_action, real_observation,
                                                          problem.agent.observation_model,
                                                          problem.agent.transition_model)
        planner.update(problem.agent, real_action, real_observation)
        _time_used += time.time() - start

        # Info and render
        _total_reward += reward
        print("==== Step %d ====" % (i+1))
        print("Action: %s" % str(real_action))
        print("Observation: %s"
              % str({objid: real_observation.objposes[objid].pose
                     for objid in real_observation.objposes
                     if (real_observation.objposes[objid].pose != ObjectObservation.NULL\
                         and objid in problem.env.target_objects)}))
        print("Reward: %s" % str(reward))
        print("Reward (Cumulative): %s" % str(total_reward))
        if isinstance(planner, pomdp_py.POUCT):
            print("__num_sims__: %d" % planner.last_num_sims)
            
        if visualize:
            viz.update(problem.agent.robot_id,
                       real_action, real_observation, problem.agent.cur_belief)
            viz.on_loop()
            viz.on_render()
            
        # Termination check
        if _find_actions_count > len(env.target_objects):
            print("FindAction limit reached.")
            break
        if _time_used > max_time:
            print("Maximum time reached.")
            break
            
# Test
def unittest():
    grid_map, robot_char = world0
    laserstr = make_laser_sensor(90, (1, 5), 0.5, False)
    problem = MosOOPOMDP(robot_char,  # r is the robot character
                         grid_map=grid_map,
                         sensors={robot_char: laserstr},
                         prior="uniform")
    solve(problem,
          max_depth=10,
          discount_factor=0.99,
          planning_time=1.,
          exploration_const=1000,
          visualize=True,
          max_time=120,
          max_steps=500)

if __name__ == "__main__":
    unittest()
