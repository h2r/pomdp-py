import pomdp_py

description = "testing framework basics"

def test_agent() -> None:
    b0 = pomdp_py.Histogram({"hungry": 0.5, "full": 0.5})
    policy = pomdp_py.UniformPolicyModel(["eat", "sleep"])
    agent = pomdp_py.Agent(b0, policy)
    assert agent.policy_model.sample(b0.random()) in ["eat", "sleep"]

def run():
    test_agent()

if __name__ == "__main__":
    run()
