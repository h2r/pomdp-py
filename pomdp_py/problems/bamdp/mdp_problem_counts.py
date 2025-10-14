import pomdp_py
from pomdp_py.utils import TreeDebugger
import random
import numpy as np
import sys
import copy


class State(pomdp_py.State):
    def __init__(self, name, counts):
        self.name = name
        self.counts = counts   # dict {action_name: (success_count, failure_count)}

    def __hash__(self):
        return hash((self.name, frozenset(self.counts.items())))

    def __eq__(self, other):
        if isinstance(other, State):
            return self.name == other.name and self.counts == other.counts
        return False

    def __str__(self):
        return self.name

    def __repr__(self):
        return "State(%s, counts=%s)" % (self.name, self.counts)


class Action(pomdp_py.Action):
    def __init__(self, name):
        self.name = name

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, Action):
            return self.name == other.name
        return False

    def __str__(self):
        return self.name

    def __repr__(self):
        return "Action(%s)" % self.name


class Observation(pomdp_py.Observation):
    def __init__(self, name, counts=None):
        self.name = name
        self.counts = counts   # dict {action_name: (success_count, failure_count)}

    def __hash__(self):
        return hash((self.name, frozenset(self.counts.items())))

    def __eq__(self, other):
        if isinstance(other, Observation):
            return self.name == other.name and self.counts == other.counts
        return False

    def __str__(self):
        return self.name

    def __repr__(self):
        return "Observation(%s)" % self.name


# Observation model
class ObservationModel(pomdp_py.ObservationModel):
    def __init__(self):
        pass

    def probability(self, observation, next_state, action):
        if observation.name == next_state.name and observation.counts == next_state.counts:
            return 1.0 
        else: 
            return 0.0

    def sample(self, next_state, action):
        return Observation(next_state.name, next_state.counts)


# Transition Model
class TransitionModel(pomdp_py.TransitionModel):
    def sample(self, state, action):
        sa_successes = state.counts[action.name][0]
        sa_failures = state.counts[action.name][1]

        if state.name == "start" and action.name == "left":
            new_counts = copy.deepcopy(state.counts)
            new_counts[action.name] = (sa_successes + 1, sa_failures)
            return State("goal", counts=new_counts)
        elif state.name == "start" and action.name == "right":
            if np.random.uniform() > 0.5:
                new_counts = copy.deepcopy(state.counts)
                new_counts[action.name] = (sa_successes + 1, sa_failures)
                return State("goal", counts=new_counts)
            else:
                new_counts = copy.deepcopy(state.counts)
                new_counts[action.name] = (sa_successes, sa_failures + 1)
                return State("start", counts=new_counts)
        else:
            new_counts = copy.deepcopy(state.counts)
            new_counts[action.name] = (sa_successes + 1, sa_failures)
            return State("goal", counts=new_counts)



# Reward Model
class RewardModel(pomdp_py.RewardModel):
    def _reward_func(self, state, action, next_state):
        if action.name == "left":
            if state.name == "start":
                return -10
            elif state.name == "goal":
                return 0
            else: 
                raise ValueError("This should never trigger (left action)")
    
        elif action.name == "right":
            if state.name == "start":
                if next_state.name == "goal":   
                    return -5 
                elif next_state.name == "start":
                    return -1
            elif state.name == "goal":
                return 0
            else:
                raise ValueError("This should never trigger (right action)")               

        else: 
            raise ValueError("This should never trigger (invalid action)")

    def sample(self, state, action, next_state):
        # deterministic
        return self._reward_func(state, action, next_state)


# Policy Model
class PolicyModel(pomdp_py.RolloutPolicy):
    """A simple policy model with uniform prior over a
    small, finite action space"""

    ACTIONS = [Action(s) for s in {"left", "right"}]

    def sample(self, state):
        return random.sample(self.get_all_actions(), 1)[0]

    def rollout(self, state, history=None):
        """Treating this PolicyModel as a rollout policy"""
        return self.sample(state)

    def get_all_actions(self, state=None, history=None):
        return PolicyModel.ACTIONS


class Problem(pomdp_py.POMDP):
    """
    In fact, creating a TigerProblem class is entirely optional
    to simulate and solve POMDPs. But this is just an example
    of how such a class can be created.
    """

    def __init__(self, init_true_state, init_belief):
        """init_belief is a Distribution."""
        agent = pomdp_py.Agent(
            init_belief,
            PolicyModel(),
            TransitionModel(),
            ObservationModel(),
            RewardModel(),
        )
        env = pomdp_py.Environment(init_true_state, TransitionModel(), RewardModel())
        super().__init__(agent, env, name="Problem")

    @staticmethod
    def create():
        init_true_state = State("start", counts={"left": (1,1), "right": (1,1)})
        init_belief = pomdp_py.Histogram(
            {State("start", counts={"left": (1,1), "right": (1,1)}): 1.0}
        )
        problem = Problem(init_true_state, init_belief)
        problem.agent.set_belief(init_belief, prior=True)
        return problem


def test_planner(problem, planner, nsteps=3, debug_tree=False):
    """
    Runs the action-feedback loop of Tiger problem POMDP

    Args:
        tiger_problem (TigerProblem): a problem instance
        planner (Planner): a planner
        nsteps (int): Maximum number of steps to run this loop.
        debug_tree (bool): True if get into the pdb with a
                           TreeDebugger created as 'dd' variable.
    """
    for i in range(nsteps):
        action = planner.plan(problem.agent)
        if debug_tree:
            from pomdp_py.utils import TreeDebugger

        print("==== Step %d ====" % (i + 1))
        print(f"True state: {problem.env.state}")
        print(f"Belief: {problem.agent.cur_belief}")
        print(f"Action: {action}")
        # There is no state transition for the tiger domain.
        # In general, the ennvironment state can be transitioned
        # using
        #
        #   reward = tiger_problem.env.state_transition(action, execute=True)
        #
        # Or, it is possible that you don't have control
        # over the environment change (e.g. robot acting
        # in real world); In that case, you could skip
        # the state transition and re-estimate the state
        # (e.g. through the perception stack on the robot).
        # reward = problem.env.reward_model.sample(
        #     problem.env.state, action, problem.env.next_state
        # )
        reward = problem.env.state_transition(action, execute=True)
        print("Reward:", reward)

        # Let's create some simulated real observation;
        # Here, we use observation based on true state for sanity
        # checking solver behavior. In general, this observation
        # should be sampled from agent's observation model, as
        #
        #    real_observation = problem.agent.observation_model.sample(problem.env.state, action)
        #
        # or coming from an external source (e.g. robot sensor
        # reading). Note that probelm.env.state stores the
        # environment state after action execution.
        real_observation = Observation(problem.env.state.name, problem.env.state.counts)
        print(">> Observation:", real_observation)
        problem.agent.update_history(action, real_observation)

        # Update the belief. If the planner is POMCP, planner.update
        # also automatically updates agent belief.
        planner.update(problem.agent, action, real_observation)
        if isinstance(planner, pomdp_py.POUCT):
            print("Num sims:", planner.last_num_sims)
            print("Plan time: %.5f" % planner.last_planning_time)

        if isinstance(problem.agent.cur_belief, pomdp_py.Histogram):
            new_belief = pomdp_py.update_histogram_belief(
                problem.agent.cur_belief,
                action,
                real_observation,
                problem.agent.observation_model,
                problem.agent.transition_model,
            )
            problem.agent.set_belief(new_belief)


def make_problem(init_state="start", init_belief=[1.0, 0.0]):
    """Convenient function to quickly build a tiger domain.
    Useful for testing"""
    problem = Problem.create()
    return problem


def main():
    init_true_state = "start"
    
    problem = make_problem(init_state=init_true_state)
    init_belief = pomdp_py.Histogram(
            {State("start", counts={"left": (1,1), "right": (1,1)}): 1.0}
        )

    # print("** Testing value iteration **")
    # vi = pomdp_py.ValueIteration(horizon=3, discount_factor=0.995)
    # test_planner(problem, vi, nsteps=10)

    # print("\n** Testing POUCT **")
    # pouct = pomdp_py.POUCT(
    #     max_depth=10,
    #     discount_factor=0.95,
    #     num_sims=4096,
    #     exploration_const=50,
    #     rollout_policy=problem.agent.policy_model,
    #     show_progress=True,
    # )
    # test_planner(problem, pouct, nsteps=10)
    # TreeDebugger(problem.agent.tree).pp

    # # Reset agent belief
    # tiger.agent.set_belief(init_belief, prior=True)
    # tiger.agent.tree = None

    print("** Testing POMCP **")
    problem.agent.set_belief(
        pomdp_py.Particles.from_histogram(init_belief, num_particles=100), prior=True
    )
    pomcp = pomdp_py.POMCP(
        max_depth=20,
        discount_factor=0.995,
        num_sims=1000,
        exploration_const=50,
        rollout_policy=problem.agent.policy_model,
        show_progress=True,
        pbar_update_interval=500,
    )
    test_planner(problem, pomcp, nsteps=10)
    # TreeDebugger(problem.agent.tree).pp


if __name__ == "__main__":
    main()
