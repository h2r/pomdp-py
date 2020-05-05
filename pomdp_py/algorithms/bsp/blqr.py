"""
Implementation of B-LQR algorithm described in

"Belief space planning assuming maximum likelihood observations"
"""

import pomdp_py
import numpy as np
from scipy import optimize

class BLQR(pomdp_py.Planner):

    def __init__(self,
                 func_sysd, func_obs, jac_sysd, jac_obs, jac_sysd_u,
                 Qlarge, L, Q, R,
                 planning_horizon=15):
        """
        To initialize the planenr of B-LQR, one needs to supply parameters
        that describe the underlying linear gaussian system, and also
        matrices that describe the cost function. Note that math symbols
        (e.g. xt, ut) are vectors or matrices represented as np.arrays. 
        The B-LQR cost function (Eq.14 in the paper):

        :math:`J(b_{t:T},u_{t:T}) = \bar{m}_T^TQ_{large}\bar{m}_T+\bar{s}_T^T\Lambda\bar{s}_T + \sum_{k=t}^{T-1}\bar{m}_k^TQ\bar{m}_k+\bar{u}_t^TR\bar{u}_t`
                    
        Args:
            func_sysd (function): f: (xt, ut) -> xt+1
            func_obs (function): g: (xt) -> zt
            jac_sysd (function): dfdx: (mt, ut) -> At
            jac_obs (function): dgdx: (mt) -> Ct
            jac_sysd_u (function): dfdu (mt, ut) -> Bt
            noise_sysd (pomdp_py.Gaussian): Gaussian system noise (Eq.12)
            noise_obs (pomdp_py.Gaussian): Gaussian observation noise
                (The covariance of this Gaussian is Wt in the equations)
            Qlarge (np.array) matrix of shape :math:`d \times d` where :math:`d`
                is the dimensionality of the state vector.
            L (np.array) matrix of shape :math:`d \times d` where :math:`d`
                is the dimensionality of the state vector. (Same as :math:`\Lambda` in the equation)
            Q (np.array) matrix of shape :math:`d \times d` where :math:`d`
                is the dimensionality of the state vector.
            R (np.array) matrix of shape :math:`c \times c` where :math:`c`
                is the dimensionality of the control vector.
            planning_horizon (int): This is the :math:`T`, the planning horizon.
        """
        self._func_sysd = func_sysd
        self._func_obs = func_obs
        self._jac_sysd = jac_sysd
        self._jac_obs = jac_obs
        self._jac_sysd = jac_sysd_u
        self._Qlarge = Qlarge
        self._L = L  # Lambda
        self._Q = Q
        self._R = R
        self._planning_horizon = planning_horizon
        self._dim_state = self._Q.shape[0]
        self._dim_control = self._R.shape[0]

        
    def ekf_update_mlo(self, bt, ut, gaussian_noise):
        """
        Performs the ekf belief update assuming maximum likelihood observation.
        This follows equations 12, 13 in the paper. It's the function :math:`F`.

        Args:
            bt (tuple): a belief point bt = (mt, Cov_t) representing the belief state.
                where mt is np.array of shape (d,) and Cov_t is np.array of shape (d,d).
                The cost function equation needs to turn Cov_t into a long vector consist
                of the columns of Cov_t stiched together.
            ut (np.array): control vector
            noise_sysd (np.array): A noise term (e.g. Gaussian noise) added to the
                system dynamics (:math:`v` in Eq.12). This array should have the sam
                dimensionality as mt.
            noise_obs_cov (np.array): The covariance matrix of the Gaussian noise of the
                observation function. This corresponds to Wt in equation 13.
        """
        # TODO: FIX
        mt, Cov_t = bt        
        At = self._jac_sysd(mt, ut)
        Ct = self._jac_obs(mt)  # based on maximum likelihood observation
        Wt = gaussian_noise.cov
        Gat = np.dot(np.dot(At, Cov_t), At.transpose())  # Gamma_t = At*Cov_t*At^T
        CGC_W_inv = np.linalg.inv(np.dot(np.dot(Ct, Gat), Ct.transpose()) + Wt)
        Cov_next = Gat - np.dot(np.dot(np.dot(np.dot(Gat, Ct.transpose()), CGC_W_inv), Ct), Gat)
        m_next = self._func_sysd(mt, ut) + gaussian_noise.random()
        return (m_next, Cov_next)

    def _b_u_seg(self, x, i):
        """Returns the elements of x that corresponds to the belief and controls,
        as a tuple (m_i, Covi, u_i)"""
        d = self._dim_state
        c = self._dim_control
        start = i*(d + (d*d) + c)
        end = (i+1)*(d + (d*d) + c)
        m_i_end = i*(d + (d*d) + c) + d
        Covi_end = i*(d + (d*d) + c) + d + d*d
        u_end = i*(d + (d*d) + c) + d + d*d + c
        return x[start:m_i_end], x[m_i_end:Covi_end], x[Covi_end:u_end]

    def _cost_func_segmented(self, x, b_des, u_des, num_segments):
        """The cost function in eq 17. This will be used as part of the objective function for
        scipy optimizer. Therefore we only take in one input vector, x.

        We require the structure of x to be:

        [ m1 | Cov 1 | u1 ] ... [ m_i | Cov i | u_i ]

        Use the _b_u_seg function to obtain the individual parts.

        Additional Arguments:
            b_des (tuple): The desired belief (mT, CovT). This is needed to compute the cost function.
            u_des (list): A list of desired controls at the beginning of each segment.
                If this information is not available, pass in an empty list.
        
        """
        if len(u_des) > 0 and len(u_des) != num_segments:
            raise ValueError("The list of desired controls, if provided, must have one control"\
                             "per segment")
        
        m_des, Cov_des = b_des
        s_des = Cov_des.transpose().reshape(-1,)  # column vectors of covariance matrix stiched together
        sLs = np.dot(np.dot(s_des.transpose(), self._L), s_des)
        Summation = 0
        for i in range(num_segments):
            m_i, _, u_i = self._b_u_seg(x, i)
            m_i_diff = m_i - m_des
            if len(u_des) > 0:
                u_i_des = u_des[i]
            else:
                u_i_des = np.zeros(self._dim_control)
            u_i_diff = u_i - u_i_des
            Summation += np.dot(np.dot(m_i_diff.transpose(), self._Q), m_i_diff)\
                + np.dot(np.dot(u_i_diff.transpose(), self._R), u_i_diff)
        return sLs + Summation

    def _belief_constraint(self, x, i, gaussian_noise, num_segments):
        """
        bi' = phi(b_i-1', u_i-1')
        """
        if i - 1 < 0:
            raise ValueError("Invalid index %d for belief constraint" % i)
        m_i, Cov_i, u_i = self._b_u_seg(x, i)
        s_i = Cov_i.transpose().reshape(-1,)
        sum_m_im1, sum_Cov_im1 = self._integrate_segment(x, i-1, gaussian_noise, num_segments)
        sum_s_im1 = sum_Cov_im1.transpose().reshape(-1,)
        return sum(m_i - sum_m_im1) + sum(s_i - sum_sm1)

    def _mean_final_constraint(self, x, m_des, num_segments):
        m_k, _, _ = self._b_u_seg(x, self._planning_horizon // num_segments-1)
        return m_k - m_des

    def _belief_constraint_mean(self, x, i, sum_m_1m1):
        m_i, _, _ = self._b_u_seg(x, i)
        return m_i - sum_m_1m1

    def _belief_constraint_cov(self, x, i, sum_s_1m1):
        _, s_i, _ = self._b_u_seg(x, i)
        return s_i - sum_s_1m1

    def _integrate_segment(self, x, i, gaussian_noise, num_segments):
        """This is to represent equation 18."""
        m_i, s_i, u_i = self._b_u_seg(x, i)
        Cov_i = s_i.reshape(self._dim_state, self._dim_state).transpose()
        b_seg = [self.ekf_update_mlo((m_i, Cov_i), u_i, gaussian_noise)]
        for t in range(1, self._planning_horizon // num_segments):
            b_seg.append(self.ekf_update_mlo(b_seg[t-1], u_i, gaussian_noise))
        # Summing gaussians is the same as summing mean and covariance separately.
        sum_mean = sum(b_seg[i][0] for i in range(len(b_seg)))
        sum_Cov = sum(b_seg[i][1] for i in range(len(b_seg)))
        return sum_mean, sum_Cov
    
    def _create_plan(self, b_0, u_0, b_des, u_des, gaussian_noise, num_segments=5):
        """Solves the SQP problem and produce beleif points and controls at segments
        Reference: https://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html"""

        # Is this correct?
        x0 = []
        m_0, Cov_0 = b_0
        s_0 = Cov_0.transpose().reshape(-1,)
        for i in range(num_segments):
            x0.extend(m_0)
            x0.extend(s_0)
            x0.extend(u_0)
        x0 = np.array(x0)
        
        constraints = []
        for i in range(num_segments):
            if i + 1 < num_segments-1:
                sum_mean_i, sum_Cov_i = self._integrate_segment(x0, i, gaussian_noise, num_segments)
                sum_s_i = sum_Cov_i.transpose().reshape(-1,)
                cons_mean = {'type': 'eq',
                             'fun': self._belief_constraint_mean,
                             'args': [i + 1, sum_mean_i]}
                cons_cov = {'type': 'eq',
                             'fun': self._belief_constraint_cov,
                             'args': [i + 1, sum_s_i]}
                constraints.append(cons_mean)
                constraints.append(cons_cov)
        cons_final = {'type': 'eq',
                      'fun': self._mean_final_constraint,
                      'args': [b_des[0], num_segments]}
        constraints.append(cons_final)

        x_res = optimize.minimize(self._cost_func_segmented, x0,
                                  method="SLSQP", args=(b_des, u_des, num_segments),
                                  constraints=constraints)
        return x_res



    

# class BLQR:

#     def __init__(self, Qlarge, L, Q, R):
#         """
#         Initialize this cost function with given constant matrices.
#         Args:
#             Qlarge (np.array) matrix of shape :math:`d \times d` where :math:`d`
#                 is the dimensionality of the state vector.
#             L (np.array) matrix of shape :math:`d \times d` where :math:`d`
#                 is the dimensionality of the state vector. (Same as :math:`\Lambda` in the equation)
#             Q (np.array) matrix of shape :math:`d \times d` where :math:`d`
#                 is the dimensionality of the state vector.
#             R (np.array) matrix of shape :math:`c \times c` where :math:`c`
#                 is the dimensionality of the control vector.
#         """
#         self._Qlarge = Qlargs
#         self._L = L
#         self._Q = Q
#         self._R = R






# def Cov_to_s(Cov):
#     """
#     Converts the covariance matrix Cov (d,d) to s, a vector (see Section II-B),
#     which is of dimension (d*d,1)
#     """
#     size = Cov.shape[0] * Cov.shape[1]
#     return Cov.transpose().reshape(size,1)  # Cov.transpose() because np array is row-major.


# def EKF_update_mlo(bt, ut, func_sysd, noise_sysd):
#     """
#     Performs the ekf belief update assuming maximum likelihood observation.
#     This follows equations 12, 13 in the paper.

#     Args:
#         bt (tuple): a belief point bt = (mt, Cov_t) representing the belief state.
#             where mt is np.array of shape (d,1) and Cov_t is np.array of shape (d,d).
#             The cost function equation needs to turn Cov_t into a long vector consist
#             of the columns of Cov_t stiched together.
#         func_sysd (function (mt, ut) -> mt+1): The function of the underlying system
#             dynamics. This corresponds to the function :math:`f` in the paper.
#         noise_sysd (np.array): A noise term (e.g. Gaussian noise) added to the
#             system dynamics. This array should have the same dimensionality as mt.
#     """
#     mt, Cov_t = bt
#     m_next = func_sysd(mt, Cov_t)
#     Cov_next = 0
    

# class BLQRCostFunction:
#     """
#     The B-LQR cost function (Eq.14 in the paper):

#     :math:`J(b_{t:T},u_{t:T}) = \bar{m}_T^TQ_{large}\bar{m}_T+\bar{s}_T^T\Lambda\bar{s}_T + \sum_{k=t}^{T-1}\bar{m}_k^TQ\bar{m}_k+\bar{u}_t^TR\bar{u}_t`
#     """
#     def __init__(self, Qlarge, L, Q, R):
#         """
#         Initialize this cost function with given constant matrices.
#         Args:
#             Qlarge (np.array) matrix of shape :math:`d \times d` where :math:`d`
#                 is the dimensionality of the state vector.
#             L (np.array) matrix of shape :math:`d \times d` where :math:`d`
#                 is the dimensionality of the state vector. (Same as :math:`\Lambda` in the equation)
#             Q (np.array) matrix of shape :math:`d \times d` where :math:`d`
#                 is the dimensionality of the state vector.
#             R (np.array) matrix of shape :math:`c \times c` where :math:`c`
#                 is the dimensionality of the control vector.
#         """
#         self._Qlarge = Qlargs
#         self._L = L
#         self._Q = Q
#         self._R = R


#     def evaluate(self, b_traj, u_traj, desired_b_traj, desired_u_traj):
#         """Compute the value of the cost function.  The inputs, with "_traj" means it
#         is a list of values over a sequence of time steps. The "desired_"
#         corresponds to the desired belief or action point (see Eq.7) on paper
#         """
#         m_traj, s_traj = [], []
#         for t in range(len(b_traj)):
#             # Each belief point b_t = (mt, Cov_t), where mt is np.array of shape (d,1)
#             # and Cov_t is np.array of shape (d,d). The cost function equation needs to
#             # turn Cov_t into a long vector consist of the columns of Cov_t stiched together.
#             mt, Cov_t = b_traj[t]
#             m_traj.append(mt)
#             st = Cov_to_s(Cov_t)
#             s_traj.append(st)
            
#         des_m_traj, des_s_traj = [], []
#         for t in range(len(desired_b_traj)):
#             des_mt, des_Cov_t = desired_b_traj[t]
#             des_m_traj.append(des_mt)
#             des_st = Cov_to_s(des_Cov_t)
#             des_s_traj.append(des_st)

#         # mT * Qlarge * mT
#         mT = m_traj[-1] - des_m_traj[-1]
#         goal_cost = np.dot(np.dot(mT.transpose(), self._Qlarge), mT)

#         # sT * L * sT
#         sT = s_traj[-1] - des_s_traj[-1]
#         direction_cost = np.dot(np.dot(sT.transpose(), self._L), sT)

#         # sum Q + R
#         departure_cost = 0
#         for t in range(len(m_traj)-1):
#             mt = m_traj[t] - des_m_traj[t]
#             ut = u_traj[t] - des_u_traj[t]
#             departure_cost +=\
#                 np.dot(np.dot(mt.transpose(), self._Q), mt)\
#                 + np.dot(np.dot(ut.transpose(), self._R), ut)
#         return goal_cost + direction_cost + departure_cost



# class BeliefSQP:
#     """
#     Defines the SQP (Sequential Quadratic Programming) problem
#     as a result of segmenting the planning horizon. Solves it
#     using scipy.optimize.minimize
#     """

