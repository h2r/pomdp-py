import pomdp_py

description = "testing framework basics"

TRANSITION = pomdp_py.TabularTransitionModel(
    {
        # state, action, next_state
        ("hungry", "eat", "full"): 0.7,
        ("hungry", "eat", "hungry"): 0.3,
        ("hungry", "sleep", "full"): 0.01,
        ("hungry", "sleep", "hungry"): 0.99,
        ("full", "eat", "full"): 0.9,
        ("full", "eat", "hungry"): 0.1,
        ("full", "sleep", "full"): 0.5,
        ("full", "sleep", "hungry"): 0.5,
    }
)


def test_agent_set_model() -> None:
    b0 = pomdp_py.Histogram({"hungry": 0.5, "full": 0.5})

    # test that agent can be created with incomplete models
    # and we can set the agent's model after its creation
    agent = pomdp_py.Agent(b0)

    agent.set_models(transition_model=TRANSITION)
    # next_state, state, action
    assert agent.transition_model.probability("full", "hungry", "eat") == 0.7

    policy = pomdp_py.UniformPolicyModel(["eat", "sleep"])
    agent.set_models(policy_model=policy)
    assert agent.policy_model.sample(b0.random()) in ["eat", "sleep"]


def test_env_set_model() -> None:
    # test that agent can be created with incomplete models
    # and we can set the agent's model after its creation
    env = pomdp_py.Environment(pomdp_py.SimpleState("hungry"))
    env.set_models(transition_model=TRANSITION)
    # next_state, state, action
    assert env.transition_model.probability("full", "hungry", "eat") == 0.7


def run() -> None:
    test_agent_set_model()
    test_env_set_model()


if __name__ == "__main__":
    run()
