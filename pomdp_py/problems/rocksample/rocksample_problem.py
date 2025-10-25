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
import argparse

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
    def __init__(self, position, rocktypes, terminal=False, removed_rocks=None):
        """
        position (tuple): (x,y) position of the rover on the grid.
        rocktypes: tuple of size k. Each is either Good or Bad.
        terminal (bool): The robot is at the terminal state.
        removed_rocks (set): set of rock IDs that have been sampled and removed.

        (It is so true that the agent's state doesn't need to involve the map!)

        x axis is horizontal. y axis is vertical.
        """
        self.position = position
        if type(rocktypes) != tuple:
            rocktypes = tuple(rocktypes)
        self.rocktypes = rocktypes
        self.terminal = terminal
        if removed_rocks is None:
            removed_rocks = set()
        self.removed_rocks = removed_rocks

    def __hash__(self):
        return hash((self.position, self.rocktypes, self.terminal, tuple(sorted(self.removed_rocks))))

    def __eq__(self, other):
        if isinstance(other, State):
            return (
                self.position == other.position
                and self.rocktypes == other.rocktypes
                and self.terminal == other.terminal
                and self.removed_rocks == other.removed_rocks
            )
        else:
            return False

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        rocks_status = []
        for rock_id in range(len(self.rocktypes)):
            if rock_id in self.removed_rocks:
                rocks_status.append("x")
            else:
                rocks_status.append(self.rocktypes[rock_id])
        return "State(%s | %s | %s)" % (
            str(self.position),
            str(rocks_status),
            str(self.terminal)
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
        next_removed_rocks = set(state.removed_rocks)  # Copy the removed rocks set
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
                    # Add the rock to removed_rocks instead of changing its type
                    next_removed_rocks.add(rock_id)
        return State(next_position, next_rocktypes, next_terminal, next_removed_rocks)

    def argmax(self, state, action):
        """Returns the most likely next state"""
        return self.sample(state, action)


class RSObservationModel(pomdp_py.ObservationModel):
    def __init__(self, rock_locs, half_efficiency_dist=20):
        self._half_efficiency_dist = half_efficiency_dist
        self._rocks = {rock_locs[pos]: pos for pos in rock_locs}

    def probability(self, observation, next_state, action):
        if isinstance(action, CheckAction):
            # Check if the rock has been removed (defensive programming)
            if action.rock_id in next_state.removed_rocks:
                # Rock has been removed, only no observation is possible
                if observation.quality is None:
                    return 1.0 - EPSILON
                else:
                    return EPSILON

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
            # Check if the rock has been removed (defensive programming)
            if action.rock_id in next_state.removed_rocks:
                # Rock has been removed, return no observation
                return Observation(None)

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
            # Check if there's a rock at current position and if it hasn't been removed
            if state.position in self._rock_locs:
                rock_id = self._rock_locs[state.position]
                # Check if rock has already been removed
                if rock_id in state.removed_rocks:
                    return -100  # Penalty for sampling an already removed rock
                # Check if rock is good
                if state.rocktypes[rock_id] == RockType.GOOD:
                    return 10
                else:
                    # Bad rock
                    return -10
            else:
                return -100  # Large penalty for sampling at non-rock position (defensive programming)

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

    def __init__(self, n, k, rock_locs=None):
        check_actions = set({CheckAction(rock_id) for rock_id in range(k)})
        self._move_actions = {MoveEast, MoveWest, MoveNorth, MoveSouth}
        self._other_actions = {SampleAction()} | check_actions
        self._all_actions = self._move_actions | self._other_actions
        self._n = n
        self._rock_locs = rock_locs if rock_locs is not None else {}

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
            motions = set(self._move_actions)
            rover_x, rover_y = state.position
            if rover_x == 0:
                motions.remove(MoveWest)
            if rover_y == 0:
                motions.remove(MoveNorth)
            if rover_y == self._n - 1:
                motions.remove(MoveSouth)

            # Filter out check actions for removed rocks and sample actions
            available_other_actions = set()
            for action in self._other_actions:
                if isinstance(action, CheckAction):
                    # Only include check actions for rocks that haven't been removed
                    if action.rock_id not in state.removed_rocks:
                        available_other_actions.add(action)
                elif isinstance(action, SampleAction):
                    # Only include SampleAction if agent is at a rock position and rock hasn't been removed
                    if state.position in self._rock_locs:
                        rock_id = self._rock_locs[state.position]
                        if rock_id not in state.removed_rocks:
                            available_other_actions.add(action)

            return list(motions | available_other_actions)

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
        init_state = State(rover_position, rocktypes, False, set())

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
            RSPolicyModel(n, k, rock_locs),
            RSTransitionModel(n, rock_locs, self.in_exit_area),
            RSObservationModel(rock_locs, half_efficiency_dist=half_efficiency_dist),
            RSRewardModel(rock_locs, self.in_exit_area),
            name=f"RockSampleAgent({n}, {k})",
        )
        env = pomdp_py.Environment(
            init_state,
            RSTransitionModel(n, rock_locs, self.in_exit_area),
            RSRewardModel(rock_locs, self.in_exit_area),
        )
        self._rock_locs = rock_locs
        super().__init__(agent, env, name="RockSampleProblem")


def test_planner(rocksample, planner, nsteps=3, discount=0.95, verbose=False):
    gamma = 1.0
    total_reward = 0
    total_discounted_reward = 0
    for i in range(nsteps):
        if verbose:
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
        if verbose:
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
        particles.append(State(init_state.position, rocktypes, False, set()))
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
    init_state = State(rover_position, rocktypes, False, set())
    belief = "uniform"
    init_belief = init_particles_belief(k, 200, init_state, belief=belief)
    rocksample = RockSampleProblem(n, k, init_state, rock_locs, init_belief, **kwargs)
    return rocksample


def calculate_std(values):
    """Calculate standard deviation of a list of values."""
    if len(values) <= 1:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return variance ** 0.5


def create_instance(n, k, **kwargs):
    init_state, rock_locs = RockSampleProblem.generate_instance(n, k)

    belief = "uniform"

    # init belief (uniform), represented in particles;
    # We don't factor the state here; We are also not doing any action prior.
    init_belief = init_particles_belief(k, 200, init_state, belief=belief)

    rocksample = RockSampleProblem(n, k, init_state, rock_locs, init_belief, **kwargs)
    return rocksample


def benchmark(verbose=False):
    k_runs = 20  # Number of runs to perform
    max_depth = 90
    num_sims = 16000
    exploration_const = 10

    print(f"*** Testing POMCP with {k_runs} runs ***")
    print(f"Max depth: {max_depth}")
    print(f"Number of simulations: {num_sims}")
    print(f"Exploration constant: {exploration_const}")

    total_rewards = []
    total_discounted_rewards = []

    for run in range(k_runs):
        print(f"\n==== Run {run + 1}/{k_runs} ====")
        print("testing with legal actions")

        # Create a fresh instance for each run
        rocksample = create_instance(11, 11)

        # Create POMCP planner
        pomcp = pomdp_py.POMCP(
            max_depth=max_depth,
            discount_factor=0.95,
            num_sims=num_sims,
            exploration_const=exploration_const,
            rollout_policy=rocksample.agent.policy_model,
            num_visits_init=1,
            show_progress=verbose
        )

        # Run the test planner
        tt, ttd = test_planner(rocksample, pomcp, nsteps=200, discount=0.95, verbose=verbose)

        total_rewards.append(tt)
        total_discounted_rewards.append(ttd)

        print(f"Run {run + 1} - Total reward: {tt}, Discounted reward: {ttd:.3f}")

    # Calculate averages
    avg_total_reward = sum(total_rewards) / len(total_rewards)
    avg_discounted_reward = sum(total_discounted_rewards) / len(total_discounted_rewards)

    print("\n" + "="*50)
    print(f"FINAL RESULTS ({k_runs} runs)")
    print("="*50)
    print(f"Average total reward: {avg_total_reward:.3f}")
    print(f"Average discounted reward: {avg_discounted_reward:.3f}")
    print(f"Standard deviation of total reward: {calculate_std(total_rewards):.3f}")
    print(f"Standard deviation of discounted reward: {calculate_std(total_discounted_rewards):.3f}")
    print(f"Min total reward: {min(total_rewards)}")
    print(f"Max total reward: {max(total_rewards)}")
    print(f"Min discounted reward: {min(total_discounted_rewards):.3f}")
    print(f"Max discounted reward: {max(total_discounted_rewards):.3f}")
    print("="*50)

def main(argv=None):
    parser = argparse.ArgumentParser(description="RockSample Problem Runner")
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run the benchmark tests for the RockSample problem.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output during the benchmark."
    )
    args = parser.parse_args(argv)

    if args.benchmark:
        benchmark(args.verbose)
    else:
        # Default behavior: run a small instance with a POMCP that returns quicker
        rocksample = create_instance(5, 5)
        pomcp = pomdp_py.POMCP(
            max_depth=30,
            discount_factor=0.95,
            planning_time=2.0,
            exploration_const=10,
            rollout_policy=rocksample.agent.policy_model,
            num_visits_init=1,
            show_progress=True
        )
        test_planner(rocksample, pomcp, nsteps=200, discount=0.95, verbose=True)


if __name__ == "__main__":
    main()
