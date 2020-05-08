from pomdp_py.representations.distribution.histogram import Histogram


def abstraction_over_histogram(current_histogram, state_mapper):
    state_mappings = {s:state_mapper(s) for s in current_histogram}
    hist = {}
    for s in current_histogram:
        a_s = state_mapper(s)
        if a_s not in hist[a_s]:
            hist[a_s] = 0
        hist[a_s] += current_histogram[s]
    return hist

def update_histogram_belief(current_histogram, 
                            real_action, real_observation,
                            observation_model, transition_model, oargs={},
                            targs={}, normalize=True, static_transition=False,
                            next_state_space=None):
    """
    update_histogram_belief(current_histogram, real_action, real_observation,
                            observation_model, transition_model, oargs={},
                            targs={}, normalize=True, deterministic=False)
    This update is based on the equation:
    :math:`B_{new}(s') = n O(z|s',a) \sum_s T(s'|s,a)B(s)`.

    Args:
        current_histogram (~pomdp_py.representations.distribution.Histogram)
            is the Histogram that represents current belief.
        real_action (~pomdp_py.framework.basics.Action)
        real_observation (~pomdp_py.framework.basics.Observation)
        observation_model (~pomdp_py.framework.basics.ObservationModel)
        transition_model (~pomdp_py.framework.basics.TransitionModel)
        oargs (dict) Additional parameters for observation_model (default {})
        targs (dict) Additional parameters for transition_model (default {})
        normalize (bool) True if the updated belief should be normalized
        static_transition (bool) True if the transition_model is treated as static;
            This basically means Pr(s'|s,a) = Indicator(s' == s). This then means
            that sum_s Pr(s'|s,a)*B(s) = B(s'), since s' and s have the same state space.
            This thus helps reduce the computation cost by avoiding the nested iteration
            over the state space; But still, updating histogram belief requires
            iteration of the state space, which may already be prohibitive.
        next_state_space (set) the state space of the updated belief. By default,
            this parameter is None and the state space given by current_histogram
            will be directly considered as the state space of the updated belief.
            This is useful for space and time efficiency in problems where the state
            space contains parts that the agent knows will deterministically update,
            and thus not keeping track of the belief over these states.

    Returns:
        Histogram: the histogram distribution as a result of the update
    """
    new_histogram = {}  # state space still the same.
    total_prob = 0
    if next_state_space is None:
        next_state_space = current_histogram
    for next_state in next_state_space:
        observation_prob = observation_model.probability(real_observation,
                                                         next_state,
                                                         real_action,
                                                         **oargs)
        if not static_transition:
            transition_prob = 0
            for state in current_histogram:
                transition_prob += transition_model.probability(next_state,
                                                                state,
                                                                real_action,
                                                                **targs) * current_histogram[state]
        else:
            transition_prob = current_histogram[next_state]
            
        new_histogram[next_state] = observation_prob * transition_prob
        total_prob += new_histogram[next_state]

    # Normalize
    if normalize:
        for state in new_histogram:
            if total_prob > 0:
                new_histogram[state] /= total_prob
    return Histogram(new_histogram)
