"""Computes the exact value of a belief state.  Should only be used
for small POMDPs with enumerable S, A, O spaces.  This can also be
used as an exact value iteration algorithm.
"""
import pomdp_py

def value(b, *args, horizon=1):
    """
    Computes the value of a POMDP at belief state b
    for a given pomdp. The `args` argument can be either
    a dictionary or a sequence of arguments. If it is a
    sequence of arguments, those should be in the same order
    as the arguments for `_value`, that is, S, A, Z, T, O, R, gamma.
    If it is a dictionary, then it is expected to contain S, A,
    Z, T, O, R, gamma as its fields.
    """
    if len(args) > 1:
        return _value(b, *args, horizon=horizon)
    else:
        pomdp = args[0]
        return _value(b, pomdp["S"], pomdp["A"], pomdp["Z"],
                      pomdp["T"], pomdp["O"], pomdp["R"], pomdp["gamma"],
                      horizon=horizon)

def _value(b, S, A, Z, T, O, R, gamma, horizon=1):
    """
    Computes the value of a POMDP at belief state b,
    given a POMDP defined by S, A, Z, T, O, R and gamma.

    Args:
        b (dict or Histogram): belief state, maps from every state in S to a probability
        S (list): A list of states
        A (list): A list of actions
        Z (list): A list of observations
        T (TransitionModel): The pomdp_py.TransitionModel where probability is defined
        O (ObservationModel): The pomdp_py.ObservationModel where probability is defined
        R (RewardModel): The pomdp_py.RewardModel: deterministic
        gamma (float): The discount factor
        horizon (int): The planning horizon (rewards are accumulated up
                       to the planning horizon).
    Returns:
        float: value at belief
    """
    assert type(S) == list, "S must be a list"
    assert type(A) == list, "A must be a list"
    assert type(Z) == list, "Z must be a list"
    max_qvalue = float('-inf')
    for a in A:
        # Compute Q(b,a)
        qv = qvalue(b, a, S, A, Z, T, O, R, gamma, horizon=horizon)
        if max_qvalue < qv:
            max_qvalue = qv
    return max_qvalue

def qvalue(b, a, S, A, Z, T, O, R, gamma, horizon=1):
    """Compute Q(v,a)"""
    r = expected_reward(b, R, a, T)

    expected_future_value = 0.0
    if horizon > 1:
        for o in Z:
            # compute Pr(o|b,a)*V(b')
            prob_o = belief_observation_model(o, b, a, T, O)
            # If o has non-zero probability
            if prob_o > 0:
                next_b = belief_update(b, a, o, T, O)
                next_value = value(next_b, S, A, Z, T, O, R, gamma,
                                   horizon=horizon-1)
                value_o = prob_o * next_value
                expected_future_value += value_o

    return r + gamma * expected_future_value

def expected_reward(b, R, a, T=None):
    """Returns the expected reward at a given belief"""
    r = 0.0
    for s in b:
        if T is not None:
            for sp in b:
                r += b[s] * R.sample(s, a, sp) * T.probability(sp, s, a)
        else:
            r += b[s] * R.sample(s, a, None)
    return r

def belief_observation_model(o, b, a, T, O):
    """Returns the probability of Pr(o|b,a)"""
    prob = 0.0
    for s in b:
        for sp in b:
            trans_prob = T.probability(sp, s, a)
            obsrv_prob = O.probability(o, sp, a)
            prob += obsrv_prob * trans_prob * b[s]
    return prob

def belief_update(b, a, o, T, O):
    """Returns the updated belief of `b` given
    action `a` and observation `o`."""
    next_b = {}
    total_prob = 0.0
    for sp in b:
        prob = O.probability(o, sp, a)
        trans_prob = 0.0
        for s in b:
            trans_prob += T.probability(sp, s, a) * b[s]
        prob *= trans_prob
        next_b[sp] = prob
        total_prob += next_b[sp]

    for s in next_b:
        # normalize
        next_b[s] /= total_prob
    return next_b


# Tests
def _create_case(noise=0.15, init_state="tiger-left"):
    """Create a tiger problem instance with uniform belief"""
    from pomdp_problems.tiger.tiger_problem import TigerProblem, TigerState
    tiger = TigerProblem(noise, TigerState(init_state),
                         pomdp_py.Histogram({TigerState("tiger-left"): 0.5,
                                             TigerState("tiger-right"): 0.5}))
    T = tiger.agent.transition_model
    O = tiger.agent.observation_model
    S = list(T.get_all_states())
    Z = list(O.get_all_observations())
    A = list(tiger.agent.policy_model.get_all_actions())
    R = tiger.agent.reward_model
    gamma = 0.95

    b0 = tiger.agent.belief
    s0 = tiger.env.state
    return b0, s0, S, A, Z, T, O, R, gamma

def _test_basic():
    b0, s0, S, A, Z, T, O, R, gamma = _create_case(noise=0.15,
                                                   init_state="tiger-left")
    horizon = 3
    qvs = {}
    for a in A:
        qvs[a.name] = qvalue(b0, a, S, A, Z, T, O, R, gamma, horizon=horizon)
    assert qvs["listen"] > qvs["open-left"]
    assert qvs["listen"] > qvs["open-right"]
    assert qvs["listen"] > qvs["stay"]
    print("Pass.")


def _test_planning():
    b0, s0, S, A, Z, T, O, R, gamma = _create_case(noise=0.15,
                                                   init_state="tiger-left")
    horizon = 3

    # Do planning with qvalue
    b = b0
    s = s0
    print("Initial state={}. Initial belief={}".format(s, b))
    _actions = []
    for step in range(10):
        a_best = None
        maxq = float('-inf')
        for a in A:
            q = qvalue(b, a, S, A, Z, T, O, R, gamma, horizon=horizon)
            if maxq < q:
                maxq = q
                a_best = a
        sp = T.sample(s, a_best)
        o = O.sample(sp, a_best)
        b = belief_update(b, a_best, o, T, O)
        print("[Step {}] action={}, value={}, observation={},  belief={}"\
              .format(step, a_best, maxq, o, b))
        _actions.append(a_best.name)
    assert "open-right" in _actions
    print("Pass.")


if __name__ == "__main__":
    _test_basic()
    _test_planning()
