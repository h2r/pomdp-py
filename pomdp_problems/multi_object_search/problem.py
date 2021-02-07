"""2D Multi-Object Search (MOS) Task.
Uses the domain, models, and agent/environment
to actually define the POMDP problem for multi-object search.
Then, solve it using POUCT or POMCP."""
import pomdp_py
from pomdp_problems.multi_object_search.env.env import *
from pomdp_problems.multi_object_search.env.visual import *
from pomdp_problems.multi_object_search.agent.agent import *
from pomdp_problems.multi_object_search.example_worlds import *
from pomdp_problems.multi_object_search.domain.observation import *
from pomdp_problems.multi_object_search.models.components.grid_map import *
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
                 sensors=None, sigma=0.01, epsilon=1,
                 belief_rep="histogram", prior={}, num_particles=100,
                 agent_has_map=False):
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
            agent_has_map (bool): If True, we assume the agent is given the occupancy
                                  grid map of the world. Then, the agent can use this
                                  map to avoid planning invalid actions (bumping into things).
                                  But this map does not help the agent's prior belief directly.

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
            dim, robots, objects, obstacles, sensors = interpret(worldstr)
            init_state = MosOOState({**objects, **robots})
            env = MosEnvironment(dim,
                          init_state, sensors,
                          obstacles=obstacles)

        # construct prior
        if type(prior) == str:
            if prior == "uniform":
                prior = {}
            elif prior == "informed":
                prior = {}
                for objid in env.target_objects:
                    groundtruth_pose = env.state.pose(objid)
                    prior[objid] = {groundtruth_pose: 1.0}

        # Potential extension: a multi-agent POMDP. For now, the environment
        # can keep track of the states of multiple agents, but a POMDP is still
        # only defined over a single agent. Perhaps, MultiAgent is just a kind
        # of Agent, which will make the implementation of multi-agent POMDP cleaner.
        robot_id = robot_id if type(robot_id) == int else interpret_robot_id(robot_id)
        grid_map = GridMap(env.width, env.length,
                           {objid: env.state.pose(objid)
                            for objid in env.obstacles}) if agent_has_map else None
        agent = MosAgent(robot_id,
                         env.state.object_states[robot_id],
                         env.target_objects,
                         (env.width, env.length),
                         env.sensors[robot_id],
                         sigma=sigma,
                         epsilon=epsilon,
                         belief_rep=belief_rep,
                         prior=prior,
                         num_particles=num_particles,
                         grid_map=grid_map)
        super().__init__(agent, env,
                         name="MOS(%d,%d,%d)" % (env.width, env.length, len(env.target_objects)))


### Belief Update ###
def belief_update(agent, real_action, real_observation, next_robot_state, planner):
    """Updates the agent's belief; The belief update may happen
    through planner update (e.g. when planner is POMCP)."""
    # Updates the planner; In case of POMCP, agent's belief is also updated.
    planner.update(agent, real_action, real_observation)

    # Update agent's belief, when planner is not POMCP
    if not isinstance(planner, pomdp_py.POMCP):
        # Update belief for every object
        for objid in agent.cur_belief.object_beliefs:
            belief_obj = agent.cur_belief.object_belief(objid)
            if isinstance(belief_obj, pomdp_py.Histogram):
                if objid == agent.robot_id:
                    # Assuming the agent can observe its own state:
                    new_belief = pomdp_py.Histogram({next_robot_state: 1.0})
                else:
                    # This is doing
                    #    B(si') = normalizer * O(oi|si',sr',a) * sum_s T(si'|s,a)*B(si)
                    #
                    # Notes: First, objects are static; Second,
                    # O(oi|s',a) ~= O(oi|si',sr',a) according to the definition
                    # of the observation model in models/observation.py.  Note
                    # that the exact belief update rule for this OOPOMDP needs to use
                    # a model like O(oi|si',sr',a) because it's intractable to
                    # consider s' (that means all combinations of all object
                    # states must be iterated).  Of course, there could be work
                    # around (out of scope) - Consider a volumetric observaiton,
                    # instead of the object-pose observation. That means oi is a
                    # set of pixels (2D) or voxels (3D). Note the real
                    # observation, oi, is most likely sampled from O(oi|s',a)
                    # because real world considers the occlusion between objects
                    # (due to full state s'). The problem is how to compute the
                    # probability of this oi given s' and a, where it's
                    # intractable to obtain s'. To this end, we can make a
                    # simplifying assumption that an object is contained within
                    # one pixel (or voxel); The pixel (or voxel) is labeled to
                    # indicate free space or object. The label of each pixel or
                    # voxel is certainly a result of considering the full state
                    # s. The occlusion can be handled nicely with the volumetric
                    # observation definition. Then that assumption can reduce the
                    # observation model from O(oi|s',a) to O(label_i|s',a) and
                    # it becomes easy to define O(label_i=i|s',a) and O(label_i=FREE|s',a).
                    # These ideas are used in my recent 3D object search work.
                    new_belief = pomdp_py.update_histogram_belief(belief_obj,
                                                                  real_action,
                                                                  real_observation.for_obj(objid),
                                                                  agent.observation_model[objid],
                                                                  agent.transition_model[objid],
                                                                  # The agent knows the objects are static.
                                                                  static_transition=objid != agent.robot_id,
                                                                  oargs={"next_robot_state": next_robot_state})
            else:
                raise ValueError("Unexpected program state."\
                                 "Are you using the appropriate belief representation?")

            agent.cur_belief.set_object_belief(objid, new_belief)


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
    random_object_belief = problem.agent.belief.object_beliefs[random_objid]
    if isinstance(random_object_belief, pomdp_py.Histogram):
        # Use POUCT
        planner = pomdp_py.POUCT(max_depth=max_depth,
                                 discount_factor=discount_factor,
                                 planning_time=planning_time,
                                 exploration_const=exploration_const,
                                 rollout_policy=problem.agent.policy_model)  # Random by default
    elif isinstance(random_object_belief, pomdp_py.Particles):
        # Use POMCP
        planner = pomdp_py.POMCP(max_depth=max_depth,
                                 discount_factor=discount_factor,
                                 planning_time=planning_time,
                                 exploration_const=exploration_const,
                                 rollout_policy=problem.agent.policy_model)  # Random by default
    else:
        raise ValueError("Unsupported object belief type %s" % str(type(random_object_belief)))

    robot_id = problem.agent.robot_id
    if visualize:
        viz = MosViz(problem.env, controllable=False)  # controllable=False means no keyboard control.
        if viz.on_init() == False:
            raise Exception("Environment failed to initialize")
        viz.update(robot_id,
                   None,
                   None,
                   None,
                   problem.agent.cur_belief)
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
        reward = problem.env.state_transition(real_action, execute=True,
                                              robot_id=robot_id)

        # Receive observation
        _start = time.time()
        real_observation = \
            problem.env.provide_observation(problem.agent.observation_model, real_action)

        # Updates
        problem.agent.clear_history()  # truncate history
        problem.agent.update_history(real_action, real_observation)
        belief_update(problem.agent, real_action, real_observation,
                      problem.env.state.object_states[robot_id],
                      planner)
        _time_used += time.time() - _start

        # Info and render
        _total_reward += reward
        if isinstance(real_action, FindAction):
            _find_actions_count += 1
        print("==== Step %d ====" % (i+1))
        print("Action: %s" % str(real_action))
        print("Observation: %s" % str(real_observation))
        print("Reward: %s" % str(reward))
        print("Reward (Cumulative): %s" % str(_total_reward))
        print("Find Actions Count: %d" %  _find_actions_count)
        if isinstance(planner, pomdp_py.POUCT):
            print("__num_sims__: %d" % planner.last_num_sims)

        if visualize:
            # This is used to show the sensing range; Not sampled
            # according to observation model.
            robot_pose = problem.env.state.object_states[robot_id].pose
            viz_observation = MosOOObservation({})
            if isinstance(real_action, LookAction) or isinstance(real_action, FindAction):
                viz_observation = \
                    problem.env.sensors[robot_id].observe(robot_pose,
                                                          problem.env.state)
            viz.update(robot_id,
                       real_action,
                       real_observation,
                       viz_observation,
                       problem.agent.cur_belief)
            viz.on_loop()
            viz.on_render()

        # Termination check
        if set(problem.env.state.object_states[robot_id].objects_found)\
           == problem.env.target_objects:
            print("Done!")
            break
        if _find_actions_count >= len(problem.env.target_objects):
            print("FindAction limit reached.")
            break
        if _time_used > max_time:
            print("Maximum time reached.")
            break

# Test
def unittest():
    # random world
    grid_map, robot_char = random_world(10, 10, 5, 10)
    laserstr = make_laser_sensor(90, (1, 4), 0.5, False)
    proxstr = make_proximity_sensor(4, False)
    problem = MosOOPOMDP(robot_char,  # r is the robot character
                         sigma=0.05,  # observation model parameter
                         epsilon=0.95, # observation model parameter
                         grid_map=grid_map,
                         sensors={robot_char: proxstr},
                         prior="uniform",
                         agent_has_map=True)
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
