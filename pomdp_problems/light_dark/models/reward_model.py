"""Defines the RewardModel for the continuous light-dark domain;
Also provides a sparse reward model, easier to be understood and
probably a "better" reward with less human-designed factors.

Origin: Belief space planning assuming maximum likelihood observations

Quote from the paper:

    The cost function (equation 8) used recurring state and action costs
    of R = diag (0.5, 0.5) and Q = diag (0.5, 0.5) and a final cost on
    covariance, Lambda=200. B-LQR had an additional large final cost on
    mean. Direct transcription used a final value constraint on mean,
    m = (0,0) instead.

More context (quote from the paper) and my comments/thoughts.

    In general, we are concerned with the problem of _reaching a given
    region of state space with high certainty_.

Interesting - I thought POMDPs are about producing a sequence of actions to
maximize the reward, which does not explicitly require the belief distribution
to end up in any particular shape or form. In robotics problems the goal is
specified through the reward; Indeed, for problems such as localization, one
type of reward is to guide its actions is based on certainty of belief in its
current position. But this is a specific kind of guided reward; A reward
function that is more general in terms of possible behavior as a consequence is
sparse goal-directed reward. This kind of reward is used in the
`POMCPOW paper <https://arxiv.org/pdf/1709.06196.pdf>`_ regarding the
light-dark domain.

    For a Gaussian belief space, this corresponds to a cost function that is minimized
    at zero covariance. However, it _may be_ more important to reduce covariance
    in some directions over others. ... We define a finite horizon quadratic cost-to-go
    function ... As a result, we re-write the cost-to-go function as:

        :math:`J(b_{\\tau:T}, u_{\\tau:T}) = s_T^{\\top}\Lambda s_T +
        \sum_{t=\\tau}^{T-1}\\tilde{m}_t^{\\top} Q \\tilde{m}_t + \\tilde{u}_t^{\\top} R \\tilde{u}_t`

    where :math:`\Lambda=\sum_{i=1}^K w_iL_i` and the `cost matrix` :math:`L_i` is
    a function of unit vectors :math:`\{\hat{n}_1,\cdots,\hat{n}_k\}` pointing in :math:`k`
    directions in which it is desried to minimize covariance and let the `relative importance`
    of these directions be described by weights :math:`w_1,\cdots,w_k`.
"""
import pomdp_py
import pomdp_problems.util as util

class LightDarkRewardModel(pomdp_py.RewardModel):
    
    def __init__(self, goal_pos, big=100, small=1):
        self._goal_pos = goal_pos
        self.big = big
        self.small = small
        
    @property
    def goal_pos(self):
        return self._goal_pos
                
class SparseRewardModel(LightDarkRewardModel):
    """
    This sparse reward function comes from the POMCPOW paper <https://arxiv.org/pdf/1709.06196.pdf>`_
    titled `Online algorithms for POMDPs with continuous state, action, and observation spaces`.
    This can be interpreted as:

    If the agent takes a "stay" action, i.e. produces a control (vx,vy) that results in
    no/little positional change, AND the agent is at the goal location, the agent receives a +100 reward.
    If the agents "stays" at another non-goal location, the agent receives a -100 reward. Each
    step also has a -1 step cost.
    
    Since the action spaces is continuous in this implementation (different from the POMCPOW paper),
    we here allow a circular tolerance zone where staying inside that zone counts as reaching the goal.
    It is straightforward to extend this to the discrete action space case in POMCPOW paper by setting
    the tolerance radius to be zero.
    """
    def __init__(self,
                 goal_pos, big=100,
                 small=1, tolerance=0.5, minimum_motion=1e-3):
        """
        Args:
            goal_pos (tuple): 2D goal location
            big (float): the heavy reward/penalty
            small (float): the small, step-wise cost.
            tolerance (float): goal tolerance radius,
            minimum_motion (float): upper bound of the change of the robot position that
               can still be regarded as "staying"
        """
        self._tolerance = tolerance
        self._mininum_motion = minimum_motion
        super().__init__(goal_pos, big=big, small=small)
        
    def probability(self, reward, state, action, next_state, normalized=False, **kwargs):
        if reward == self._reward_func(state, action, next_state):
            return 1.0
        else:
            return 0.0
        
    def sample(self, state, action, next_state,
               normalized=False, robot_id=None):
        # deterministic
        return self._reward_func(state, action, next_state, robot_id=robot_id)
    
    def argmax(self, state, action, next_state, normalized=False, robot_id=None):
        """Returns the most likely reward"""
        return self._reward_func(state, action, next_state, robot_id=robot_id)

    def _reward_func(self, state, action, next_state):
        reward = -self.small
        dx = next_state.position[0] - state.position[0]
        dy = next_state.position[y] - state.position[y]
        if abs(dx) <= self._minimum_motion and abs(dy) <= self._minimum_motion:
            # agent is staying            
            if util.euclidean_dist(next_state.position, self._goal_pos) < self._tolerance:
                # close to goal
                reward += self._big
            else:
                # not close to goal
                reward -= self._big
        return reward

            
