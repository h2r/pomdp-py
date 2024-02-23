import pomdp_py

description = "testing framework basics"

def test_agent() -> None:
    b0 = pomdp_py.Histogram({"hungry": 0.5, "full": 0.5})
    policy = pomdp_py.UniformPolicyModel(["eat", "sleep"])

    # test that agent can be created with incomplete models
    agent = pomdp_py.Agent(b0, policy)
    assert agent.policy_model.sample(b0.random()) in ["eat", "sleep"]

    # test that
    transition = pomdp_py.TabularTransitionModel(
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
    agent.set_models(transition_model=transition)
    # next_state, state, action
    assert agent.transition_model.probability("full", "hungry", "eat") == 0.7



def run():
    test_agent()

if __name__ == "__main__":
    run()
