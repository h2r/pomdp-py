
cdef class Distribution:
    """
    A Distribution is a probability function that maps
    from variable value to a real value.
    """
    def __getitem__(self, varval):
        raise NotImplemented
    def __setitem__(self, varval, value):
        raise NotImplemented
    def __hash__(self):
        return id(self)
    def __eq__(self, other):
        return id(self) == id(other)
    def __iter__(self):
        """Initialization of iterator over the values in this distribution"""
        raise NotImplemented
    def __next__(self):
        """Returns the next value of the iterator"""
        raise NotImplemented
    def probability(self, varval):
        return self[varval]

cdef class GenerativeDistribution(Distribution):
    def argmax(self, **kwargs):
        return self.mpe(**kwargs)
    def mpe(self, **kwargs):
        raise NotImplemented
    def random(self, **kwargs):
        # Sample a state based on the underlying belief distribution
        raise NotImplemented
    def get_histogram(self):
        """Returns a dictionary from state to probability"""
        raise NotImplemented

cdef class ObservationModel:
    def probability(self, observation, next_state, action, **kwargs):
        raise NotImplemented
    def sample(self, next_state, action, **kwargs):
        """Returns observation"""
        raise NotImplemented
    def argmax(self, next_state, action, **kwargs):
        """Returns the most likely observation"""
        raise NotImplemented
    def get_distribution(self, next_state, action, **kwargs):
        """Returns the underlying distribution of the model"""
        raise NotImplemented
    def get_all_observations(self):
        """Returns a set of all possible observations, if feasible."""
        raise NotImplemented        
    
cdef class TransitionModel:
    
    def probability(self, next_state, state, action, **kwargs):
        raise NotImplemented
        
    def sample(self, state, action, **kwargs):
        """Returns next_state"""
        raise NotImplemented
    def argmax(self, state, action, **kwargs):
        """Returns the most likely next state"""
        raise NotImplemented
    def get_distribution(self, state, action, **kwargs):
        """Returns the underlying distribution of the model"""
        raise NotImplemented
    def get_all_states(self):
        """Returns a set of all possible states, if feasible."""
        raise NotImplemented

cdef class RewardModel:
    
    def probability(self, reward, state, action, next_state, **kwargs):
        raise NotImplemented
        
    def sample(self, state, action, next_state, **kwargs):
        """Returns a reward"""
        raise NotImplemented
    
    def argmax(self, state, action, next_state, **kwargs):
        """Returns the most likely reward"""
        raise NotImplemented
    def get_distribution(self, state, action, next_state, **kwargs):
        """Returns the underlying distribution of the model"""
        raise NotImplemented    

cdef class BlackboxModel:
    
    def sample(self, state, action, **kwargs):
        """Sample (s',o,r) ~ G(s',o,r)"""
        raise NotImplemented
    
    def argmax(self, state, action, **kwargs):
        """Returns the most likely (s',o,r)"""
        raise NotImplemented

cdef class PolicyModel:
    """The reason to have a policy model is to accommodate problems
    with very large action spaces, and the available actions may vary
    depending on the state (that is, certain actions have probabilty=0)"""
    
    def probability(self, action, state, **kwargs):
        raise NotImplemented
        
    def sample(self, state, **kwargs):
        """Returns an action"""
        raise NotImplemented
    
    def argmax(self, state, **kwargs):
        """Returns the most likely reward"""
        raise NotImplemented
    def get_distribution(self, state, **kwargs):
        """Returns the underlying distribution of the model"""
        raise NotImplemented
    def get_all_actions(self, *args, **kwargs):
        """Returns a set of all possible actions, if feasible."""
        raise NotImplemented
    def update(self, state, next_state, action, **kwargs):
        """Policy model may be updated given a (s,a,s') pair."""
        pass
    
# Belief distribution is just a distribution. There's nothing special,
# except that the update/abstraction function can be performed over them.
# But it would make the class hierarchy a lot more complicated if belief
# distribution is also made explicit, which means, for example, a belief
# distribution represented as a histogram would have to do multiple
# inheritance; doing so, the additional value is little.


"""Because T, R, O may be different for the agent versus the environment,
it does not make much sense to have the POMDP class to hold this information;
instead, Agent should have its own T, R, O, pi and the Environment should
have its own T, R. The job of a POMDP is only to verify whether a given state,
action, or observation are valid."""

cdef class POMDP:
    def __init__(self, agent, env, name="POMDP"):
        self.agent = agent
        self.env = env

cdef class Action:
    def __eq__(self, other):
        raise NotImplemented
    def __hash__(self):
        raise NotImplemented        
cdef class State:
    def __eq__(self, other):
        raise NotImplemented
    def __hash__(self):
        raise NotImplemented        

cdef class Observation:
    def __eq__(self, other):
        raise NotImplemented
    def __hash__(self):
        raise NotImplemented        

cdef class Agent:
    def __init__(self, init_belief,
                 policy_model,
                 transition_model=None,
                 observation_model=None,
                 reward_model=None,
                 blackbox_model=None):
        self._init_belief = init_belief
        self._policy_model = policy_model
        
        self._transition_model = transition_model
        self._observation_model = observation_model
        self._reward_model = reward_model
        self._blackbox_model = blackbox_model
        # It cannot be the case that both explicit models and blackbox model are None.
        if self._blackbox_model is None:
            assert self._transition_model is not None\
                and self._observation_model is not None\
                and self._reward_model is not None

        # For online planning
        self._cur_belief = init_belief
        self._history = ()

    @property
    def history(self):
        # history are of the form ((a,o),...);
        return self._history

    def update_history(self, real_action, real_observation):
        self._history += ((real_action, real_observation),)

    @property
    def init_belief(self):
        return self._init_belief

    @property
    def belief(self):
        return self.cur_belief

    @property
    def cur_belief(self):
        return self._cur_belief

    def set_belief(self, belief, prior=False):
        self._cur_belief = belief
        if prior:
            self._init_belief = belief

    def sample_belief(self):
        return self._cur_belief.random()

    @property
    def observation_model(self):
        return self._observation_model

    @property
    def transition_model(self):
        return self._transition_model

    @property
    def reward_model(self):
        return self._reward_model

    @property
    def policy_model(self):
        return self._policy_model

    @property
    def blackbox_model(self):
        return self._blackbox_model

    @property
    def generative_model(self):
        return self.blackbox_model

    def add_attr(self, attr_name, attr_value):
        """A function that allows adding attributes to the agent.
        Sometimes useful for planners to store agent-specific information."""
        if hasattr(self, attr_name):
            raise ValueError("attributes %s already exists for agent." % attr_name)
        else:
            setattr(self, attr_name, attr_value)

    def update(self, real_action, real_observation, **kwargs):
        """updates the history and performs belief update"""
        raise NotImplemented

    @property
    def all_states(self):
        """Only available if the transition model implements
        `get_all_states`."""
        return self.transition_model.get_all_states()

    @property
    def all_actions(self):
        """Only available if the policy model implements
        `get_all_actions`."""
        return self.policy_model.get_all_actions()

    @property
    def all_observations(self):
        """Only available if the observation model implements
        `get_all_observations`."""
        return self.observation_model.get_all_observations()

    def valid_actions(self, *args, **kwargs):
        return self.policy_model.get_all_actions(*args, **kwargs)


cdef class Environment:
    """An Environment maintains the true state of the world.
    For example, it is the 2D gridworld, rendered by pygame.
    Or it could be the 3D simulated world rendered by OpenGL.
    Therefore, when coding up an Environment, the developer
    should have in mind how to represent the state so that
    it can be used by a POMDP or OOPOMDP.

    The Environment is passive. It never observes nor acts.
    """
    def __init__(self, init_state,
                 transition_model,
                 reward_model):
        self._init_state = init_state
        self._transition_model = transition_model
        self._reward_model = reward_model
        self._cur_state = init_state

    @property
    def state(self):
        return self.cur_state

    @property
    def cur_state(self):
        return self._cur_state

    @property
    def transition_model(self):
        return self._transition_model

    @property
    def reward_model(self):
        return self._reward_model
    
    def state_transition(self, action, execute=True, **kwargs):
        """Modifies current state of the environment"""
        next_state, reward, _ = sample_explict_models(self.transition_model, None, self.reward_model,
                                                      self.state, action,
                                                      discount_factor=kwargs.get("discount_factor", 1.0))
        if execute:
            self._cur_state = next_state
        return reward

    def provide_observation(self, observation_model, action, **kwargs):
        return observation_model.sample(self.state, action, **kwargs)
    
    
cdef class Option(Action):
    """An option is a temporally abstracted action that
    is defined by (I, pi, B), where I is a initiation set,
    pi is a policy, and B is a termination condition

    Described in ``Between MDPs and semi-MDPs:
    A framework for temporal abstraction in reinforcement learning''
    """
    def initiation(self, state, **kwargs):
        """Returns True if the given parameters satisfy the initiation set"""
        raise NotImplemented

    def termination(self, state, **kwargs):
        """Returns a boolean of whether state satisfies
        the termination condition"""
        raise NotImplemented

    @property
    def policy(self):
        """Returns the policy model (PolicyModel) of this option."""
        raise NotImplemented

    def sample(self, state, **kwargs):
        """Samples an action from this option's policy.
        Convenience function; Can be overriden if don't
        feel like writing a PolicyModel class"""
        return self.policy.sample(state, **kwargs)
    
    def __eq__(self, other):
        raise NotImplemented
    
    def __hash__(self):
        raise NotImplemented

    # Remark in Sutton etal'99:
    # Note that even if a policy is Markov and all of the options
    # it selects are Markov, the corresponding flat policy is unlikely to be
    # Markov if any of the options are multi-step (temporally extended).
    #    --> that's why planning with options defined over an MDP
    #        results in a semi-MDP.


cpdef sample_generative_model(Agent agent, State state, Action action, float discount_factor=1.0):
    """
    (s', o, r) ~ G(s, a)

    Returns (s', o, r, n_steps)
    """
    cdef tuple result

    if agent.transition_model is None:
        result = agent.generative_model.sample(state, action)
    else:
        result = sample_explict_models(agent.transition_model,
                                       agent.observation_model,
                                       agent.reward_model,
                                       state,
                                       action,
                                       discount_factor)
    return result


cpdef sample_explict_models(TransitionModel T, ObservationModel O, RewardModel R,
                            State state, Action action, float discount_factor=1.0):
    cdef State next_state
    cdef Observation observation
    cdef float reward
    cdef Option option
    cdef int nsteps = 0

    if isinstance(action, Option):
        # The action is an option; simulate a rollout of the option
        option = action
        if not option.initiation(state):
            # state is not in the initiation set of the option. This is
            # similar to the case when you are in a particular (e.g. terminal)
            # state and certain action cannot be performed, which will still
            # be added to the PO-MCTS tree because some other state with the
            # same history will allow this action. In this case, that certain
            # action will lead to no state change, no observation, and 0 reward,
            # because nothing happened.
            if O is not None:
                return state, None, 0, 0
            else:
                return state, 0, 0

        reward = 0
        step_discount_factor = 1.0
        while not option.termination(state):
            action = option.sample(state)
            next_state = T.sample(state, action)
            # For now, we don't care about intermediate observations (future work?).
            reward += step_discount_factor * R.sample(state, action, next_state)
            step_discount_factor *= discount_factor
            state = next_state
            nsteps += 1
        # sample observation at the end, where action is the last action.
        # (doesn't quite make sense to just use option as the action at this point.)
    else:
        next_state = T.sample(state, action)
        reward = R.sample(state, action, next_state)
        nsteps += 1
    if O is not None:
        observation = O.sample(next_state, action)
        return next_state, observation, reward, nsteps
    else:
        return next_state, reward, nsteps
    
