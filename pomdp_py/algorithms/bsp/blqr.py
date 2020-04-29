"""
Implementation of B-LQR algorithm described in

"Belief space planning assuming maximum likelihood observations"
"""

import numpy as np

def Sigma_to_s(Sigma):
    """
    Converts the covariance matrix Sigma (d,d) to s, a vector (see Section II-B),
    which is of dimension (d*d,1)
    """
    size = Sigma.shape[0] * Sigma.shape[1]
    return Sigma.transpose().reshape(size,1)  # Sigma.transpose() because np array is row-major.


class BLQRCostFunction:
    """
    The B-LQR cost function (Eq.14 in the paper):

    :math:`J(b_{t:T},u_{t:T}) = \bar{m}_T^TQ_{large}\bar{m}_T+\bar{s}_T^T\Lambda\bar{s}_T + \sum_{k=t}^{T-1}\bar{m}_k^TQ\bar{m}_k+\bar{u}_t^TR\bar{u}_t`
    """
    def __init__(self, Qlarge, L, Q, R):
        """
        Initialize this cost function with given constant matrices.
        Args:
            Qlarge (np.array) matrix of shape :math:`d \times d` where :math:`d`
                is the dimensionality of the state vector.
            L (np.array) matrix of shape :math:`d \times d` where :math:`d`
                is the dimensionality of the state vector. (Same as :math:`\Lambda` in the equation)
            Q (np.array) matrix of shape :math:`d \times d` where :math:`d`
                is the dimensionality of the state vector.
            R (np.array) matrix of shape :math:`c \times c` where :math:`c`
                is the dimensionality of the control vector.
        """
        self._Qlarge = Qlargs
        self._L = L
        self._Q = Q
        self._R = R


    def evaluate(self, b_traj, u_traj, desired_b_traj, desired_u_traj):
        """Compute the value of the cost function.  The inputs, with "_traj" means it
        is a list of values over a sequence of time steps. The "desired_"
        corresponds to the desired belief or action point (see Eq.7) on paper
        """
        m_traj, s_traj = [], []
        for t in range(len(b_traj)):
            # Each belief point b_t = (mt, Sigma_t), where mt is np.array of shape (d,1)
            # and Sigma_t is np.array of shape (d,d). The cost function equation needs to
            # turn Sigma_t into a long vector consist of the columns of Sigma_t stiched together.
            mt, Sigma_t = b_traj[t]
            m_traj.append(mt)
            st = Sigma_to_s(Sigma_t)
            s_traj.append(st)
            
        des_m_traj, des_s_traj = [], []
        for t in range(len(desired_b_traj)):
            des_mt, des_Sigma_t = desired_b_traj[t]
            des_m_traj.append(des_mt)
            des_st = Sigma_to_s(des_Sigma_t)
            des_s_traj.append(des_st)

        # mT * Qlarge * mT
        mT = m_traj[-1] - des_m_traj[-1]
        goal_cost = np.dot(np.dot(mT.transpose(), self._Qlarge), mT)

        # sT * L * sT
        sT = s_traj[-1] - des_s_traj[-1]
        direction_cost = np.dot(np.dot(sT.transpose(), self._L), sT)

        # sum Q + R
        departure_cost = 0
        for t in range(len(m_traj)-1):
            mt = m_traj[t] - des_m_traj[t]
            ut = u_traj[t] - des_u_traj[t]
            departure_cost +=\
                np.dot(np.dot(mt.transpose(), self._Q), mt)\
                + np.dot(np.dot(ut.transpose(), self._R), ut)
        return goal_cost + direction_cost + departure_cost
