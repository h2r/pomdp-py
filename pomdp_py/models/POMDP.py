# Python imports.
from collections import defaultdict

# Other imports.
from simple_rl.mdp.MDPClass import MDP

class POMDP(MDP):
    ''' Abstract class for a Partially Observable Markov Decision Process. '''

    def __init__(self, actions, transition_func, reward_func, observation_func,
                 init_belief, init_true_state=None, gamma=0.99, step_cost=0):
        '''
        In addition to the input parameters needed to define an MDP, the POMDP
        definition requires an observation function, a way to update the belief
        state and an initial belief. Note that the `reward_func` should be the
        function that can be used by the planner, that is, it computes reward
        by the agent internally. This is different from the reward provided by
        the environment after the agent executes a real action; this reward should
        be specified in the `env_reward_func` under `execute_agent_action_update_belief`.
        
        Args:
            actions (list)
            observations (list)
            transition_func: T(s, a) -> s'
            reward_func: R(s, a) -> float
            observation_func: O(s, a) -> z
            init_belief (defaultdict): initial probability distribution over states
            belief_updater_type (str): discrete/kalman/particle
            gamma (float)
            step_cost (int)
        '''
        self.observation_func = observation_func
        self.init_belief = init_belief
        self.cur_belief = init_belief
        self._last_real_action = None
        MDP.__init__(self, actions, transition_func, reward_func,
                     init_state=init_true_state,
                     gamma=gamma, step_cost=step_cost)

    def get_cur_belief(self):
        return self.cur_belief

    def get_observation_func(self):
        '''
        Returns:
            observation_function: O(s, a) -> o
        '''
        return self.observation_func

    # DEPRECATED!
    def execute_agent_action_update_belief(self, action, **kwargs):
        # Execute agent action AND update the current belief. This function is used
        # by planners (e.g. POMCP) and it combines the two steps to ensure flexibility
        # of how the reward is computed in the POMDP.
        def env_reward_func(*params):
            """reward provided by the environment after the agent executes a real action"""
            raise NotImplemented
        raise NotImplemented


    # The Environment should execute actions and maintain true states. The
    # POMDP is only used to provide observation and reward, based on the
    # O and R specified in the POMDP; The transition function does not
    # matter when a real action is being executed, because a real next
    # state will occur.
    def real_action_taken(self, real_action, state, next_state):
        observation = self.observation_func(next_state, real_action)
        # Note: This may not be the environment's reward function. 
        reward = self.reward_func(state, real_action, next_state)
        self._last_real_action = real_action
        return observation, reward

    def belief_update(self, real_action, real_observation, **kwargs):
        raise NotImplemented

