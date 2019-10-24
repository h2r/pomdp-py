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

    # Deprecated!
    def plan_and_execute_next_action(self):
        action = self.plan_next_action()
        return self.execute_next_action(action)

    # Deprecated!    
    def execute_next_action(self, action):
        reward, observation = self._pomdp.execute_agent_action_update_belief(action)
        return action, reward, observation

    def plan_next_action(self):
        return random.choice(self._pomdp.actions)

    def update(self, real_action, real_observation):
        """update the planner (e.g. truncate the search tree) after
        a real action has been taken and a real observation obtained"""
        pass

    @property
    def params(self):
        """returns a dictionary of parameter names to values"""
        return {}




    
