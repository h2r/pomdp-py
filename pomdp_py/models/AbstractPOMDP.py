from pomdp_py import POMDP, State
import copy


class AbstractPOMDP(POMDP):

    SEARCH = "search"
    BACKTRACK = "backtrack"

    def __init__(self,
                 abstract_actions,
                 abstract_transition_func,
                 abstract_reward_func,
                 abstract_observation_func,
                 init_abstract_belief,
                 init_abstract_state,
                 gamma=0.99):
        super().__init__(abstract_actions,
                         abstract_transition_func,
                         abstract_reward_func,
                         abstract_observation_func,
                         init_abstract_belief,
                         init_abstract_state,
                         gamma=gamma)

    def state_mapper(self, state, *params, **kwargs):
        """maps a given state to an abstract state"""
        raise NotImplemented

    def action_mapper(self, abstract_action, *params, **kwargs):
        """a function that maps a given abstract action to a list of
        lower-level actions"""
        raise NotImplemented

    def observation_mapper(self, observations, *params, **kwargs):
        """a function that maps lower-level observations into an
        abstract observation"""
        raise NotImplemented

    def generate_pomdp_from_abstract_action(self, abstract_action, *params, **kwargs):
        """Generates a POMDP object based on the abstract action taken. Intuitively,
        this means, after taking the action 'go to kitchen', a POMDP of searching
        within the kitchen should then be generated.

        It is REQUIRED that the generated POMDP shares the same world object as
        the abstract POMDP, such that when actions are executed by the POMDP,
        the real world is changed for the abstract POMDP as well."""
        raise NotImplemented
    
    def execute_agent_action_update_belief(self, action, **kwargs):
        # Execute agent action AND update the current belief. This function is used
        # by planners (e.g. POMCP) and it combines the two steps to ensure flexibility
        # of how the reward is computed in the POMDP.
        def env_reward_func(*params):
            """reward provided by the environment after the agent executes a real action"""
            raise NotImplemented
        raise NotImplemented

    def belief_update(self, real_action, real_observation, **kwargs):
        raise NotImplemented

def reshape_distribution(distribution, state_mapping_func):
    if isinstance(distribution, BeliefDistribution_Histogram):
        new_histogram = {}
        total_prob = 0
        for state in distribution:
            abstract_state = state_mapping_func(state)
            if abstract_state not in new_histogram:
                new_histogram[abstract_state] = 0
            new_histogram[abstract_state] += distribution[state]
            total_prob += distribution[state]
        for abstract_state in new_histogram:
            new_histogram[abstract_state] /= total_prob
        distribution_copy = copy.deepcopy(distribution)
        distribution_copy._histogram = new_histogram
    elif isinstance(distribution, BeliefDistribution_Particles):
        new_particles = [state_mapping_func(s) for s in distribution.particles]
        distribution_copy = copy.deepcopy(distribution)
        distribution_copy._particles = new_particles
    return distribution_copy
        
        
            
