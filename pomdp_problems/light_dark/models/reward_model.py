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

"""
