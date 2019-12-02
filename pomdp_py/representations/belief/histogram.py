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

def update_histogram_belief(current_histogram, real_action, real_observation,
                            observation_model, transition_model,
                            oargs={}, targs={}, normalize=True):
    """
    This update is based on the equation:
    B_new(s') = n * O(z|s',a) sum_s T(s'|s,a)B(s).

    `current_histogram` is the Histogram that represents current belief.

    Returns the histogram distribution as a result of the update
    """
    new_histogram = {}  # state space still the same.
    total_prob = 0
    for next_state in current_histogram:
        observation_prob = observation_model.probability(real_observation,
                                                         next_state,
                                                         real_action,
                                                         **oargs)
        transition_prob = 0
        for state in current_histogram:
            transition_prob += transition_model.probability(next_state,
                                                            state,
                                                            real_action,
                                                            **targs) * current_histogram[state]
        new_histogram[next_state] = observation_prob * transition_prob
        total_prob += new_histogram[next_state]

    # Normalize
    if normalize:
        for state in new_histogram:
            if total_prob > 0:
                new_histogram[state] /= total_prob
    return Histogram(new_histogram)
