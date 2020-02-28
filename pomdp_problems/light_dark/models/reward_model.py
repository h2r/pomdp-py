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

class RewardModel(pomdp_py.RewardModel):

    def __init__(self):
        pass
