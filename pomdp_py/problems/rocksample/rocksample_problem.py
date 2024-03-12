"""RockSample(n,k) problem

Origin: Heuristic Search Value Iteration for POMDPs (UAI 2004)

Description:

State space:

    Position {(1,1),(1,2),...(n,n)}
    :math:`\\times` RockType_1 :math:`\\times` RockType_2, ..., :math:`\\times` RockType_k
    where RockType_i = {Good, Bad}
    :math:`\\times` TerminalState

    (basically, the positions of rocks are known to the robot,
     but not represented explicitly in the state space. Check_i
     will smartly check the rock i at its location.)

Action space:

    North, South, East, West, Sample, Check_1, ..., Check_k
    The first four moves the agent deterministically
    Sample: samples the rock at agent's current location
    Check_i: receives a noisy observation about RockType_i
    (noise determined by eta (:math:`\eta`). eta=1 -> perfect sensor; eta=0 -> uniform)

Observation: observes the property of rock i when taking Check_i.  The
     observation may be noisy, depending on an efficiency parameter which
     decreases exponentially as the distance increases between the rover and
     rock i. 'half_efficiency_dist' influences this parameter (larger, more robust)

Reward: +10 for Sample a good rock. -10 for Sampling a bad rock.
        Move to exit area +10. Other actions have no cost or reward.

Initial belief: every rock has equal probability of being Good or Bad.

"""

import pomdp_py
import random
import math
import numpy as np
import sys
import copy

EPSILON = 1e-9


def euclidean_dist(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


class RockType:
    GOOD = "good"
    BAD = "bad"

    @staticmethod
    def invert(rocktype):
        if rocktype == "good":
            return "bad"
        else:
            return "good"
        # return 1 - rocktype

    @staticmethod
    def random(p=0.5):
        if random.uniform(0, 1) >= p:
            return RockType.GOOD
        else:
            return RockType.BAD


class State(pomdp_py.State):
    def __init__(self, position, rocktypes, terminal=False):
        """
        position (tuple): (x,y) position of the rover on the grid.
        rocktypes: tuple of size k. Each is either Good or Bad.
        terminal (bool): The robot is at the terminal state.

        (It is so true that the agent's state doesn't need to involve the map!)

        x axis is horizontal. y axis is vertical.
        """
        self.position = position
        if type(rocktypes) != tuple:
            rocktypes = tuple(rocktypes)
        self.rocktypes = rocktypes
        self.terminal = terminal

    def __hash__(self):
        return hash((self.position, self.rocktypes, self.terminal))

    def __eq__(self, other):
        if isinstance(other, State):
            return (
                self.position == other.position
                and self.rocktypes == other.rocktypes
                and self.terminal == other.terminal
            )
        else:
            return False

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return "State(%s | %s | %s)" % (
            str(self.position),
            str(self.rocktypes),
            str(self.terminal),
        )


class Action(pomdp_py.Action):
    def __init__(self, name):
        self.name = name

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, Action):
            return self.name == other.name
        elif type(other) == str:
            return self.name == other

    def __str__(self):
        return self.name

    def __repr__(self):
        return "Action(%s)" % self.name


class MoveAction(Action):
    EAST = (1, 0)  # x is horizontal; x+ is right. y is vertical; y+ is up.
    WEST = (-1, 0)
    NORTH = (0, -1)
    SOUTH = (0, 1)

    def __init__(self, motion, name):
        if motion not in {
            MoveAction.EAST,
            MoveAction.WEST,
            MoveAction.NORTH,
            MoveAction.SOUTH,
        }:
            raise ValueError("Invalid move motion %s" % motion)
        self.motion = motion
        super().__init__("move-%s" % str(name))


MoveEast = MoveAction(MoveAction.EAST, "EAST")
MoveWest = MoveAction(MoveAction.WEST, "WEST")
MoveNorth = MoveAction(MoveAction.NORTH, "NORTH")
MoveSouth = MoveAction(MoveAction.SOUTH, "SOUTH")


class SampleAction(Action):
    def __init__(self):
        super().__init__("sample")


class CheckAction(Action):
    def __init__(self, rock_id):
        self.rock_id = rock_id
        super().__init__("check-%d" % self.rock_id)


class Observation(pomdp_py.Observation):
    def __init__(self, quality):
        self.quality = quality

    def __hash__(self):
        return hash(self.quality)

    def __eq__(self, other):
        if isinstance(other, Observation):
            return self.quality == other.quality
        elif type(other) == str:
            return self.quality == other

    def __str__(self):
        return str(self.quality)

    def __repr__(self):
        return "Observation(%s)" % str(self.quality)


class RSTransitionModel(pomdp_py.TransitionModel):
    """The model is deterministic"""

    def __init__(self, n, rock_locs, in_exit_area):
        """
        rock_locs: a map from (x,y) location to rock_id
        in_exit_area: a function (x,y) -> Bool that returns True if (x,y) is in exit area
        """
        self._n = n
        self._rock_locs = rock_locs
        self._in_exit_area = in_exit_area

    def _move_or_exit(self, position, action):
        expected = (position[0] + action.motion[0], position[1] + action.motion[1])
        if self._in_exit_area(expected):
            return expected, True
        else:
            return (
                max(0, min(position[0] + action.motion[0], self._n - 1)),
                max(0, min(position[1] + action.motion[1], self._n - 1)),
            ), False

    def probability(self, next_state, state, action, normalized=False, **kwargs):
        if next_state != self.sample(state, action):
            return EPSILON
        else:
            return 1.0 - EPSILON

    def sample(self, state, action):
        next_position = tuple(state.position)
        rocktypes = tuple(state.rocktypes)
        next_rocktypes = rocktypes
        next_terminal = state.terminal
        if state.terminal:
            next_terminal = True  # already terminated. So no state transition happens
        else:
            if isinstance(action, MoveAction):
                next_position, exiting = self._move_or_exit(state.position, action)
                if exiting:
                    next_terminal = True
            elif isinstance(action, SampleAction):
                if next_position in self._rock_locs:
                    rock_id = self._rock_locs[next_position]
                    _rocktypes = list(rocktypes)
                    _rocktypes[rock_id] = RockType.BAD
                    next_rocktypes = tuple(_rocktypes)
        return State(next_position, next_rocktypes, next_terminal)

    def argmax(self, state, action):
        """Returns the most likely next state"""
        return self.sample(state, action)


class RSObservationModel(pomdp_py.ObservationModel):
    def __init__(self, rock_locs, half_efficiency_dist=20):
        self._half_efficiency_dist = half_efficiency_dist
        self._rocks = {rock_locs[pos]: pos for pos in rock_locs}

    def probability(self, observation, next_state, action):
        if isinstance(action, CheckAction):
            # compute efficiency
            rock_pos = self._rocks[action.rock_id]
            dist = euclidean_dist(rock_pos, next_state.position)
            eta = (1 + pow(2, -dist / self._half_efficiency_dist)) * 0.5

            # compute probability
            actual_rocktype = next_state.rocktypes[action.rock_id]
            if actual_rocktype == observation:
                return eta
            else:
                return 1.0 - eta
        else:
            if observation.quality is None:
                return 1.0 - EPSILON  # expected to receive no observation
            else:
                return EPSILON

    def sample(self, next_state, action, argmax=False):
        if not next_state.terminal and isinstance(action, CheckAction):
            # compute efficiency
            rock_pos = self._rocks[action.rock_id]
            dist = euclidean_dist(rock_pos, next_state.position)
            eta = (1 + pow(2, -dist / self._half_efficiency_dist)) * 0.5

            if argmax:
                keep = eta > 0.5
            else:
                keep = eta > random.uniform(0, 1)

            actual_rocktype = next_state.rocktypes[action.rock_id]
            if not keep:
                observed_rocktype = RockType.invert(actual_rocktype)
                return Observation(observed_rocktype)
            else:
                return Observation(actual_rocktype)
        else:
            # Terminated or not a check action. So no observation.
            return Observation(None)

        return self._probs[next_state][action][observation]

    def argmax(self, next_state, action):
        """Returns the most likely observation"""
        return self.sample(next_state, action, argmax=True)


class RSRewardModel(pomdp_py.RewardModel):
    def __init__(self, rock_locs, in_exit_area):
        self._rock_locs = rock_locs
        self._in_exit_area = in_exit_area

    def sample(self, state, action, next_state, normalized=False, **kwargs):
        # deterministic
        if state.terminal:
            return 0  # terminated. No reward
        if isinstance(action, SampleAction):
            # need to check the rocktype in `state` because it has turned bad in `next_state`
            if state.position in self._rock_locs:
                if state.rocktypes[self._rock_locs[state.position]] == RockType.GOOD:
                    return 10
                else:
                    # No rock or bad rock
                    return -10
            else:
                return 0  # problem didn't specify penalty for sampling empty space.

        elif isinstance(action, MoveAction):
            if self._in_exit_area(next_state.position):
                return 10
        return 0

    def argmax(self, state, action, next_state, normalized=False, **kwargs):
        raise NotImplementedError

    def probability(
        self, reward, state, action, next_state, normalized=False, **kwargs
    ):
        raise NotImplementedError


class RSPolicyModel(pomdp_py.RolloutPolicy):
    """Simple policy model according to problem description."""

    def __init__(self, n, k):
        check_actions = set({CheckAction(rock_id) for rock_id in range(k)})
        self._move_actions = {MoveEast, MoveWest, MoveNorth, MoveSouth}
        self._other_actions = {SampleAction()} | check_actions
        self._all_actions = self._move_actions | self._other_actions
        self._n = n

    def sample(self, state, normalized=False, **kwargs):
        return random.sample(self.get_all_actions(state=state), 1)[0]

    def probability(self, action, state, normalized=False, **kwargs):
        raise NotImplementedError

    def argmax(self, state, normalized=False, **kwargs):
        """Returns the most likely reward"""
        raise NotImplementedError

    def get_all_actions(self, **kwargs):
        state = kwargs.get("state", None)
        if state is None:
            return list(self._all_actions)
        else:
            motions = set(self._all_actions)
            rover_x, rover_y = state.position
            if rover_x == 0:
                motions.remove(MoveWest)
            if rover_y == 0:
                motions.remove(MoveNorth)
            if rover_y == self._n - 1:
                motions.remove(MoveSouth)
            return list(motions | self._other_actions)

    def rollout(self, state, history=None):
        return random.sample(self.get_all_actions(state=state), 1)[0]


class RockSampleProblem(pomdp_py.POMDP):
    @staticmethod
    def random_free_location(n, not_free_locs):
        """returns a random (x,y) location in nxn grid that is free."""
        while True:
            loc = (random.randint(0, n - 1), random.randint(0, n - 1))
            if loc not in not_free_locs:
                return loc

    def in_exit_area(self, pos):
        return pos[0] == self._n

    @staticmethod
    def generate_instance(n, k):
        """Returns init_state and rock locations for an instance of RockSample(n,k)"""

        rover_position = (0, random.randint(0, n - 1))
        rock_locs = {}  # map from rock location to rock id
        for i in range(k):
            loc = RockSampleProblem.random_free_location(
                n, set(rock_locs.keys()) | set({rover_position})
            )
            rock_locs[loc] = i

        # random rocktypes
        rocktypes = tuple(RockType.random() for i in range(k))

        # Ground truth state
        init_state = State(rover_position, rocktypes, False)

        return init_state, rock_locs

    def print_state(self):
        string = "\n______ID______\n"
        rover_position = self.env.state.position
        rocktypes = self.env.state.rocktypes
        # Rock id map
        for y in range(self._n):
            for x in range(self._n + 1):
                char = "."
                if x == self._n:
                    char = ">"
                if (x, y) in self._rock_locs:
                    char = str(self._rock_locs[(x, y)])
                if (x, y) == rover_position:
                    char = "R"
                string += char
            string += "\n"
        string += "_____G/B_____\n"
        # Good/bad map
        for y in range(self._n):
            for x in range(self._n + 1):
                char = "."
                if x == self._n:
                    char = ">"
                if (x, y) in self._rock_locs:
                    if rocktypes[self._rock_locs[(x, y)]] == RockType.GOOD:
                        char = "$"
                    else:
                        char = "x"
                if (x, y) == rover_position:
                    char = "R"
                string += char
            string += "\n"
        print(string)

    def __init__(
        self, n, k, init_state, rock_locs, init_belief, half_efficiency_dist=20
    ):
        self._n, self._k = n, k
        agent = pomdp_py.Agent(
            init_belief,
            RSPolicyModel(n, k),
            RSTransitionModel(n, rock_locs, self.in_exit_area),
            RSObservationModel(rock_locs, half_efficiency_dist=half_efficiency_dist),
            RSRewardModel(rock_locs, self.in_exit_area),
        )
        env = pomdp_py.Environment(
            init_state,
            RSTransitionModel(n, rock_locs, self.in_exit_area),
            RSRewardModel(rock_locs, self.in_exit_area),
        )
        self._rock_locs = rock_locs
        super().__init__(agent, env, name="RockSampleProblem")


def test_planner(rocksample, planner, nsteps=3, discount=0.95):
    gamma = 1.0
    total_reward = 0
    total_discounted_reward = 0
    for i in range(nsteps):
        print("==== Step %d ====" % (i + 1))
        action = planner.plan(rocksample.agent)
        # pomdp_py.visual.visualize_pouct_search_tree(rocksample.agent.tree,
        #                                             max_depth=5, anonymize=False)

        true_state = copy.deepcopy(rocksample.env.state)
        env_reward = rocksample.env.state_transition(action, execute=True)
        true_next_state = copy.deepcopy(rocksample.env.state)

        real_observation = rocksample.env.provide_observation(
            rocksample.agent.observation_model, action
        )
        rocksample.agent.update_history(action, real_observation)
        planner.update(rocksample.agent, action, real_observation)
        total_reward += env_reward
        total_discounted_reward += env_reward * gamma
        gamma *= discount
        print("True state: %s" % true_state)
        print("Action: %s" % str(action))
        print("Observation: %s" % str(real_observation))
        print("Reward: %s" % str(env_reward))
        print("Reward (Cumulative): %s" % str(total_reward))
        print("Reward (Cumulative Discounted): %s" % str(total_discounted_reward))
        if isinstance(planner, pomdp_py.POUCT):
            print("__num_sims__: %d" % planner.last_num_sims)
            print("__plan_time__: %.5f" % planner.last_planning_time)
        if isinstance(planner, pomdp_py.PORollout):
            print("__best_reward__: %d" % planner.last_best_reward)
        print("World:")
        rocksample.print_state()

        if rocksample.in_exit_area(rocksample.env.state.position):
            break
    return total_reward, total_discounted_reward


def init_particles_belief(k, num_particles, init_state, belief="uniform"):
    num_particles = 200
    particles = []
    for _ in range(num_particles):
        if belief == "uniform":
            rocktypes = []
            for i in range(k):
                rocktypes.append(RockType.random())
            rocktypes = tuple(rocktypes)
        elif belief == "groundtruth":
            rocktypes = copy.deepcopy(init_state.rocktypes)
        particles.append(State(init_state.position, rocktypes, False))
    init_belief = pomdp_py.Particles(particles)
    return init_belief


def minimal_instance(**kwargs):
    # A particular instance for debugging purpose
    n, k = 2, 2
    rover_position = (0, 0)
    rock_locs = {}  # map from rock location to rock id
    rock_locs[(0, 1)] = 0
    rock_locs[(1, 1)] = 1
    rocktypes = ("good", "good")
    # Ground truth state
    init_state = State(rover_position, rocktypes, False)
    belief = "uniform"
    init_belief = init_particles_belief(k, 200, init_state, belief=belief)
    rocksample = RockSampleProblem(n, k, init_state, rock_locs, init_belief, **kwargs)
    return rocksample


def create_instance(n, k, **kwargs):
    init_state, rock_locs = RockSampleProblem.generate_instance(n, k)

    belief = "uniform"

    # init belief (uniform), represented in particles;
    # We don't factor the state here; We are also not doing any action prior.
    init_belief = init_particles_belief(k, 200, init_state, belief=belief)

    rocksample = RockSampleProblem(n, k, init_state, rock_locs, init_belief, **kwargs)
    return rocksample


def main():
    rocksample = debug_instance()  # create_instance(7, 8)
    rocksample.print_state()

    print("*** Testing POMCP ***")
    pomcp = pomdp_py.POMCP(
        max_depth=30,
        discount_factor=0.95,
        num_sims=10000,
        exploration_const=5,
        rollout_policy=rocksample.agent.policy_model,
        num_visits_init=1,
    )
    tt, ttd = test_planner(rocksample, pomcp, nsteps=100, discount=0.95)


if __name__ == "__main__":
    main()
