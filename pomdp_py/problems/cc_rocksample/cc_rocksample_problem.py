from __future__ import annotations
import copy
import math
import pomdp_py
from pomdp_py.problems.rocksample.rocksample_problem import (
    RockSampleProblem,
    create_instance,
    RSTransitionModel,
    RSObservationModel,
    RSPolicyModel,
    CheckAction,
    RSRewardModel,
    init_particles_belief,
    State,
)


class RSResponse(pomdp_py.Response):

    def __init__(self, reward: int = 0, cost: int = 0) -> None:
        super().__init__()
        self.reward = int(reward)
        self.cost = int(cost)

    def __add__(self, other: RSResponse) -> RSResponse:
        return RSResponse(self.reward + other.reward, self.cost + other.cost)

    def __mul__(self, other: float | int) -> RSResponse:
        return RSResponse(self.reward * other, self.cost * other)

    def __str__(self) -> str:
        return f"reward={self.reward}, cost={self.cost}"

    @staticmethod
    def null() -> RSResponse:
        return RSResponse(reward=0, cost=0)


class RSCostModel(pomdp_py.CostModel):

    def sample(
        self,
        state: pomdp_py.State,
        action: pomdp_py.Action,
        next_state: pomdp_py.State,
        reward: float = 0,
        **kwargs,
    ) -> int:
        cost = 0
        if reward < 0:
            cost += 1
        if isinstance(action, CheckAction):
            cost += 1
        return cost


class RSResponseModel(pomdp_py.ResponseModel):
    def __init__(
        self,
        reward_model: RSRewardModel,
        cost_model: RSCostModel,
    ):
        super().__init__()
        self._reward_model = reward_model
        self._cost_model = cost_model

    def null_response(self) -> RSResponse:
        return RSResponse(reward=0, cost=0)

    def sample(
        self, state: pomdp_py.State, action: pomdp_py.Action, next_state: pomdp_py.State
    ) -> RSResponse:
        reward = self._reward_model.sample(
            state=state, action=action, next_state=next_state
        )
        cost = self._cost_model.sample(
            state=state, action=action, next_state=next_state, reward=reward
        )
        return RSResponse(reward, cost)


class CCRockSampleProblem(RockSampleProblem):

    def __init__(
        self,
        n_grid: int,
        n_rocks: int,
        init_state: State,
        rock_locs: dict[tuple[int, int], int],
        init_belief: pomdp_py.GenerativeDistribution,
        half_efficiency_dist: int = 20,
    ):
        super().__init__(
            n=n_grid,
            k=n_rocks,
            init_state=init_state,
            rock_locs=rock_locs,
            init_belief=init_belief,
            half_efficiency_dist=half_efficiency_dist,
        )

    def build_agent(
        self,
        n: int,
        k: int,
        rock_locs: dict[tuple[int, int], int],
        init_belief: pomdp_py.GenerativeDistribution,
        half_efficiency_dist: int,
    ) -> pomdp_py.ResponseAgent:
        return pomdp_py.ResponseAgent(
            init_belief=init_belief,
            policy_model=RSPolicyModel(n, k),
            transition_model=RSTransitionModel(n, rock_locs, self.in_exit_area),
            observation_model=RSObservationModel(
                rock_locs, half_efficiency_dist=half_efficiency_dist
            ),
            response_model=RSResponseModel(
                reward_model=RSRewardModel(rock_locs, self.in_exit_area),
                cost_model=RSCostModel(),

            ),
        )

    def build_env(
        self, n: int, init_state: State, rock_locs: dict[tuple[int, int], int]
    ) -> pomdp_py.ResponseEnvironment:
        return pomdp_py.ResponseEnvironment(
            init_state=init_state,
            transition_model=RSTransitionModel(n, rock_locs, self.in_exit_area),
            response_model=RSResponseModel(
                reward_model=RSRewardModel(rock_locs, self.in_exit_area),
                cost_model=RSCostModel(),
            ),
        )


def test_planner(
    cc_rocksample: CCRockSampleProblem,
    ccpomcp: pomdp_py.CCPOMCP,
    nsteps: int = 3,
    discount: float = 0.95,
):
    gamma: float = 1.0
    total_response = RSResponse.null()
    total_discounted_response = RSResponse.null()

    for i in range(nsteps):
        print("==== Step %d ====" % (i + 1))
        action = ccpomcp.plan(cc_rocksample.agent)

        true_state = copy.deepcopy(cc_rocksample.env.state)
        env_response = cc_rocksample.env.state_transition(action, execute=True)

        real_observation = cc_rocksample.env.provide_observation(
            cc_rocksample.agent.observation_model, action
        )
        cc_rocksample.agent.update_history(action, real_observation)
        ccpomcp.update(cc_rocksample.agent, action, real_observation)
        total_response += env_response
        total_discounted_response += (env_response * gamma)
        gamma *= discount

        print("True state: %s" % true_state)
        print("Action: %s" % str(action))
        print("Observation: %s" % str(real_observation))
        print("Response: %s" % str(env_response))
        print("Response (Cumulative): %s" % str(total_response))
        print("Response (Cumulative Discounted): %s" % str(total_discounted_response))
        print("__num_sims__: %d" % ccpomcp.last_num_sims)
        print("__plan_time__: %.5f" % ccpomcp.last_planning_time)
        print("World:")
        cc_rocksample.print_state()

        if cc_rocksample.in_exit_area(cc_rocksample.env.state.position):
            break
    return total_response, total_discounted_response


def create_instance(n_grid: int, n_rocks: int) -> CCRockSampleProblem:
    init_state, rock_locs = CCRockSampleProblem.generate_instance(n_grid, n_rocks)
    belief = "uniform"
    init_belief = init_particles_belief(n_rocks, 200, init_state, belief=belief)
    return CCRockSampleProblem(
        n_grid=n_grid,
        n_rocks=n_rocks,
        init_state=init_state,
        rock_locs=rock_locs,
        init_belief=init_belief,
    )


def main(n_grid: int = 7, n_rocks: int = 8) -> None:
    cc_rocksample = create_instance(n_grid=n_grid, n_rocks=n_rocks)
    cc_rocksample.print_state()

    k_discount_factor = 0.95
    k_max_depth = int(math.log(0.001) / math.log(k_discount_factor))
    k_max_reward = 10
    k_min_reward = -10

    print("*** Testing CC-POMCP ***")
    ccpomcp = pomdp_py.CCPOMCP(
        r_diff=float(k_max_reward - k_min_reward),
        alpha_n=1.0 / len(cc_rocksample.agent.cur_belief),
        nu=1.0,
        tau=1.0,
        cost_constraint=1.0,
        max_depth=k_max_depth,
        discount_factor=k_discount_factor,
        num_sims=10000,
        exploration_const=20,
        rollout_policy=cc_rocksample.agent.policy_model,
        num_visits_init=1,
    )
    total_response, total_discounted_response = test_planner(
        cc_rocksample=cc_rocksample,
        ccpomcp=ccpomcp,
        nsteps=100,
        discount=k_discount_factor,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ngrid", type=int, default=7)
    parser.add_argument("--nrocks", type=int, default=8)
    args = parser.parse_args()

    main(n_grid=args.ngrid, n_rocks=args.nrocks)
