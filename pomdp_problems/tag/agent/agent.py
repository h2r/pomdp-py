import pomdp_py
import copy
import random
from pomdp_problems.tag.domain.observation import *
from pomdp_problems.tag.domain.action import *
from pomdp_problems.tag.domain.state import *
from pomdp_problems.tag.models.observation_model import *
from pomdp_problems.tag.models.transition_model import *
from pomdp_problems.tag.models.reward_model import *
from pomdp_problems.tag.models.policy_model import *
from pomdp_problems.tag.models.components.motion_policy import *
from pomdp_problems.tag.models.components.grid_map import *

## initialize belief
def initialize_belief(grid_map, init_robot_position, prior={}):
    """Initialize belief.
    
    Args:
        grid_map (GridMap): Holds information of the map occupancy
        prior (dict): A map from (x,y)->[0,1]. If empty, the belief
            will be uniform."""
    hist = {}  # state -> prob
    total_prob = 0.0
    for target_position in prior:
        state = TagState(init_robot_position, target_position, False)
        hist[state] = prior[target_position]
        total_prob += hist[state]

    for x in range(grid_map.width):
        for y in range(grid_map.length):
            if (x,y) in grid_map.obstacle_poses:
                # Skip obstacles
                continue
            state = TagState(init_robot_position, (x,y), False)
            if len(prior) > 0:
                if (x,y) not in prior:
                    hist[state] = 1e-9
            else:
                hist[state] = 1.0
                total_prob += hist[state]
    # Normalize
    for state in hist:
        hist[state] /= total_prob

    hist_belief = pomdp_py.Histogram(hist)
    return hist_belief

def initialize_particles_belief(grid_map, init_robot_position, num_particles=100, prior={}):
    """Initialize belief.
    
    Args:
        grid_map (GridMap): Holds information of the map occupancy
        prior (dict): A map from (x,y)->[0,1]. If empty, the belief
            will be uniform."""
    particles = []
    if len(prior) > 0:
        # prior knowledge provided. Just use the prior knowledge
        prior_sum = sum(prior[pose] for pose in prior)
        for pose in prior[objid]:
            state = TagState(init_robot_position, pose)
            amount_to_add = (prior[objid][pose] / prior_sum) * num_particles
            for _ in range(amount_to_add):
                particles.append(state)    
    else:
        while len(particles) < num_particles:
            target_position = (random.randint(0, grid_map.width-1),
                               random.randint(0, grid_map.length-1))
            if target_position in grid_map.obstacle_poses:
                # Skip obstacles
                continue            
            state = TagState(init_robot_position, target_position, False)
            particles.append(state)
    return pomdp_py.Particles(particles)


## belief update
def belief_update(agent, real_action, real_observation):
    # Update agent belief
    current_mpe_state = agent.cur_belief.mpe()
    next_robot_position = agent.transition_model.sample(current_mpe_state, real_action).robot_position
    
    next_state_space = set({})
    for state in agent.cur_belief:
        next_state = copy.deepcopy(state)
        next_state.robot_position = next_robot_position
        next_state_space.add(next_state)

    new_belief = pomdp_py.update_histogram_belief(
        agent.cur_belief, real_action, real_observation,
        agent.observation_model, agent.transition_model,
        next_state_space=next_state_space)
    
    agent.set_belief(new_belief)

class TagAgent(pomdp_py.Agent):
    
    def __init__(self,
                 init_belief,
                 grid_map,
                 pr_stay=0.2,
                 small=1,
                 big=10):
        self._grid_map = grid_map        
        target_motion_policy = TagTargetMotionPolicy(grid_map,
                                                     pr_stay)        
        transition_model = TagTransitionModel(grid_map,
                                              target_motion_policy)
        reward_model = TagRewardModel(small=small, big=big)
        observation_model = TagObservationModel()
        policy_model = TagPolicyModel(grid_map=grid_map)
        super().__init__(init_belief,
                         policy_model,
                         transition_model=transition_model,
                         observation_model=observation_model,
                         reward_model=reward_model)

    def clear_history(self):
        """Custum function; clear history"""
        self._history = None
        
        
