from pomdp_py import Planner, BeliefDistribution
import random

class RandomPlanner(Planner):

    class DummyDistribution(BeliefDistribution):
        def __init__(self):
            super().__init__()
        def mpe(self):
            return None
        def random(self):
            return None
        def get_histogram(self):
            return {}
        def update(self, real_action, real_observation, pomdp, **kwargs):
            return self
        def __str__(self):
            return "DummyDistribution"
    
    def __init__(self, pomdp, name="random"):
        super().__init__(pomdp, name=name, init_default_fields=False)
        self._pomdp = pomdp

    def plan_and_execute_next_action(self):
        action = random.choice(self._pomdp.actions)
        return self.execute_next_action(action)

    def execute_next_action(self, action):
        reward, observation = self._pomdp.execute_agent_action_update_belief(action)
        return action, reward, observation


