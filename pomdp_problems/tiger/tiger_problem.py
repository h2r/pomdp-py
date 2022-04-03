"""The classic Tiger problem.

This is a POMDP problem; Namely, it specifies both
the POMDP (i.e. state, action, observation space)
and the T/O/R for the agent as well as the environment.

The description of the tiger problem is as follows: (Quote from
`POMDP: Introduction to Partially Observable Markov Decision Processes
<https://cran.r-project.org/web/packages/pomdp/vignettes/POMDP.pdf>`_ by
Kamalzadeh and Hahsler )

A tiger is put with equal probability behind one
of two doors, while treasure is put behind the other one.
You are standing in front of the two closed doors and
need to decide which one to open. If you open the door
with the tiger, you will get hurt (negative reward).
But if you open the door with treasure, you receive
a positive reward. Instead of opening a door right away,
you also have the option to wait and listen for tiger noises. But
listening is neither free nor entirely accurate. You might hear the
tiger behind the left door while it is actually behind the right
door and vice versa.

States: tiger-left, tiger-right
Actions: open-left, open-right, listen
Rewards:
    +10 for opening treasure door. -100 for opening tiger door.
    -1 for listening.
Observations: You can hear either "tiger-left", or "tiger-right".

Note that in this example, the TigerProblem is a POMDP that
also contains the agent and the environment as its fields. In
general this doesn't need to be the case. (Refer to more
complicated examples.)
"""

import pomdp_py
from pomdp_py.utils import TreeDebugger
import random
import numpy as np
import sys

class TigerState(pomdp_py.State):
    def __init__(self, name):
        self.name = name
    def __hash__(self):
        return hash(self.name)
    def __eq__(self, other):
        if isinstance(other, TigerState):
            return self.name == other.name
        return False
    def __str__(self):
        return self.name
    def __repr__(self):
        return "TigerState(%s)" % self.name
    def other(self):
        if self.name.endswith("left"):
            return TigerState("tiger-right")
        else:
            return TigerState("tiger-left")

class TigerAction(pomdp_py.Action):
    def __init__(self, name):
        self.name = name
    def __hash__(self):
        return hash(self.name)
    def __eq__(self, other):
        if isinstance(other, TigerAction):
            return self.name == other.name
        return False
    def __str__(self):
        return self.name
    def __repr__(self):
        return "TigerAction(%s)" % self.name

class TigerObservation(pomdp_py.Observation):
    def __init__(self, name):
        self.name = name
    def __hash__(self):
        return hash(self.name)
    def __eq__(self, other):
        if isinstance(other, TigerObservation):
            return self.name == other.name
        return False
    def __str__(self):
        return self.name
    def __repr__(self):
        return "TigerObservation(%s)" % self.name

# Observation model
class ObservationModel(pomdp_py.ObservationModel):
    def __init__(self, noise=0.15):
        self.noise = noise

    def probability(self, observation, next_state, action):
        if action.name == "listen":
            # heard the correct growl
            if observation.name == next_state.name:
                return 1.0 - self.noise
            else:
                return self.noise
        else:
            return 0.5

    def sample(self, next_state, action):
        if action.name == "listen":
            thresh = 1.0 - self.noise
        else:
            thresh = 0.5

        if random.uniform(0,1) < thresh:
            return TigerObservation(next_state.name)
        else:
            return TigerObservation(next_state.other().name)

    def get_all_observations(self):
        """Only need to implement this if you're using
        a solver that needs to enumerate over the observation space
        (e.g. value iteration)"""
        return [TigerObservation(s)
                for s in {"tiger-left", "tiger-right"}]

# Transition Model
class TransitionModel(pomdp_py.TransitionModel):
    def probability(self, next_state, state, action):
        """According to problem spec, the world resets once
        action is open-left/open-right. Otherwise, stays the same"""
        if action.name.startswith("open"):
            return 0.5
        else:
            if next_state.name == state.name:
                return 1.0 - 1e-9
            else:
                return 1e-9

    def sample(self, state, action):
        if action.name.startswith("open"):
            return random.choice(self.get_all_states())
        else:
            return TigerState(state.name)

    def get_all_states(self):
        """Only need to implement this if you're using
        a solver that needs to enumerate over the observation space (e.g. value iteration)"""
        return [TigerState(s) for s in {"tiger-left", "tiger-right"}]

# Reward Model
class RewardModel(pomdp_py.RewardModel):
    def _reward_func(self, state, action):
        if action.name == "open-left":
            if state.name == "tiger-right":
                return 10
            else:
                return -100
        elif action.name == "open-right":
            if state.name == "tiger-left":
                return 10
            else:
                return -100
        else: # listen
            return -1

    def sample(self, state, action, next_state):
        # deterministic
        return self._reward_func(state, action)

# Policy Model
class PolicyModel(pomdp_py.RolloutPolicy):
    """A simple policy model with uniform prior over a
       small, finite action space"""
    ACTIONS = {TigerAction(s)
              for s in {"open-left", "open-right", "listen"}}

    def sample(self, state):
        return random.sample(self.get_all_actions(), 1)[0]

    def rollout(self, state, history=None):
        """Treating this PolicyModel as a rollout policy"""
        return self.sample(state)

    def get_all_actions(self, state=None, history=None):
        return PolicyModel.ACTIONS


class TigerProblem(pomdp_py.POMDP):
    """
    In fact, creating a TigerProblem class is entirely optional
    to simulate and solve POMDPs. But this is just an example
    of how such a class can be created.
    """

    def __init__(self, obs_noise, init_true_state, init_belief):
        """init_belief is a Distribution."""
        agent = pomdp_py.Agent(init_belief,
                               PolicyModel(),
                               TransitionModel(),
                               ObservationModel(obs_noise),
                               RewardModel())
        env = pomdp_py.Environment(init_true_state,
                                   TransitionModel(),
                                   RewardModel())
        super().__init__(agent, env, name="TigerProblem")

    @staticmethod
    def create(state="tiger-left", belief=0.5, obs_noise=0.15):
        """
        Args:
            state (str): could be 'tiger-left' or 'tiger-right';
                         True state of the environment
            belief (float): Initial belief that the target is
                            on the left; Between 0-1.
            obs_noise (float): Noise for the observation
                               model (default 0.15)
        """
        init_true_state = TigerState(state)
        init_belief = pomdp_py.Histogram({
            TigerState("tiger-left"): belief,
            TigerState("tiger-right"): 1.0 - belief
        })
        tiger_problem = TigerProblem(obs_noise,
                                     init_true_state, init_belief)
        tiger_problem.agent.set_belief(init_belief, prior=True)
        return tiger_problem




def test_planner(tiger_problem, planner, nsteps=3,
                 debug_tree=False):
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
        action = planner.plan(tiger_problem.agent)
        if debug_tree:
            from pomdp_py.utils import TreeDebugger
            dd = TreeDebugger(tiger_problem.agent.tree)
            import pdb; pdb.set_trace()

        print("==== Step %d ====" % (i+1))
        print("True state:", tiger_problem.env.state)
        print("Belief:", tiger_problem.agent.cur_belief)
        print("Action:", action)
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
        reward = tiger_problem.env.reward_model.sample(tiger_problem.env.state, action, None)
        print("Reward:", reward)

        # Let's create some simulated real observation;
        # Update the belief Creating true observation for
        # sanity checking solver behavior. In general, this
        # observation should be sampled from agent's observation
        # model, as
        #
        #    real_observation = tiger_problem.agent.observation_model.sample(tiger_problem.env.state, action)
        #
        # or coming from an external source (e.g. robot sensor
        # reading). Note that tiger_problem.env.state stores the
        # environment state after action execution.
        real_observation = TigerObservation(tiger_problem.env.state.name)
        print(">> Observation:",  real_observation)
        tiger_problem.agent.update_history(action, real_observation)

        # If the planner is POMCP, planner.update also updates agent belief.
        planner.update(tiger_problem.agent, action, real_observation)
        if isinstance(planner, pomdp_py.POUCT):
            print("Num sims:", planner.last_num_sims)
            print("Plan time: %.5f" % planner.last_planning_time)

        if isinstance(tiger_problem.agent.cur_belief,
                      pomdp_py.Histogram):
            new_belief = pomdp_py.update_histogram_belief(
                tiger_problem.agent.cur_belief,
                action, real_observation,
                tiger_problem.agent.observation_model,
                tiger_problem.agent.transition_model)
            tiger_problem.agent.set_belief(new_belief)

        if action.name.startswith("open"):
            # Make it clearer to see what actions are taken
            # until every time door is opened.
            print("\n")

def main():
    init_true_state = random.choice([TigerState("tiger-left"),
                                     TigerState("tiger-right")])
    init_belief = pomdp_py.Histogram({TigerState("tiger-left"): 0.5,
                                      TigerState("tiger-right"): 0.5})
    tiger_problem = TigerProblem(0.15,  # observation noise
                                 init_true_state, init_belief)

    print("** Testing value iteration **")
    vi = pomdp_py.ValueIteration(horizon=3, discount_factor=0.95)
    test_planner(tiger_problem, vi, nsteps=3)

    # Reset agent belief
    tiger_problem.agent.set_belief(init_belief, prior=True)

    print("\n** Testing POUCT **")
    pouct = pomdp_py.POUCT(max_depth=3, discount_factor=0.95,
                           num_sims=4096, exploration_const=50,
                           rollout_policy=tiger_problem.agent.policy_model,
                           show_progress=True)
    test_planner(tiger_problem, pouct, nsteps=10)
    TreeDebugger(tiger_problem.agent.tree).pp

    # Reset agent belief
    tiger_problem.agent.set_belief(init_belief, prior=True)
    tiger_problem.agent.tree = None

    print("** Testing POMCP **")
    tiger_problem.agent.set_belief(pomdp_py.Particles.from_histogram(init_belief, num_particles=100), prior=True)
    pomcp = pomdp_py.POMCP(max_depth=3, discount_factor=0.95,
                           num_sims=1000, exploration_const=50,
                           rollout_policy=tiger_problem.agent.policy_model,
                           show_progress=True, pbar_update_interval=500)
    test_planner(tiger_problem, pomcp, nsteps=10)
    TreeDebugger(tiger_problem.agent.tree).pp

if __name__ == '__main__':
    main()
