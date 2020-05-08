"""Implementation of B-LQR algorithm described in "Belief space planning
assuming maximum likelihood observations" :cite:`platt2010belief`
"""

import pomdp_py
import numpy as np
from scipy import optimize

class BLQR(pomdp_py.Planner):

    def __init__(self,
                 func_sysd, func_obs, jac_sysd, jac_obs, jac_sysd_u,
                 noise_obs, noise_sysd,
                 Qlarge, L, Q, R,
                 planning_horizon=15):
        """To initialize the planenr of B-LQR, one needs to supply parameters
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
            noise_obs (function): (xt) -> pomdp_py.Gaussian. Potentially
                state-dependent observation noise
            noise_sysd (function): (xt) -> pomdp_py.Gaussian. Potentially
                state-dependent system dynamics noise
            Qlarge (np.array) matrix of shape :math:`d \times d` where :math:`d`
                is the dimensionality of the state vector.
            L (np.array) matrix of shape :math:`d \times d` where :math:`d`
                is the dimensionality of the state vector.
                (Same as :math:`\Lambda` in the equation)
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
        self._noise_obs = noise_obs
        self._noise_sysd = noise_sysd        
        self._Qlarge = Qlarge
        self._L = L  # Lambda
        self._Q = Q
        self._R = R
        self._planning_horizon = planning_horizon
        self._dim_state = self._Q.shape[0]
        self._dim_control = self._R.shape[0]

        
    def ekf_update_mlo(self, bt, ut):
        """
        Performs the ekf belief update assuming maximum likelihood observation.
        This follows equations 12, 13 in the paper. It's the function :math:`F`.

        Args:
            bt (tuple): a belief point bt = (mt, Cov_t) representing the belief state.
                where mt is np.array of shape (d,) and Cov_t is np.array of shape (d,d).
                The cost function equation needs to turn Cov_t into a long vector consist
                of the columns of Cov_t stiched together.
            ut (np.array): control vector
            noise_t (pomdp_py.Gaussian): The Gaussian noise with "possibly state-dependent"
                covariance matrix Wt.
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
        Wt = self._noise_obs(mt).cov
        Gat = np.dot(np.dot(At, Cov_t), At.transpose())  # Gamma_t = At*Cov_t*At^T
        CGC_W_inv = np.linalg.inv(np.dot(np.dot(Ct, Gat), Ct.transpose()) + Wt)
        Cov_next = Gat - np.dot(np.dot(np.dot(np.dot(Gat, Ct.transpose()), CGC_W_inv), Ct), Gat)
        m_next = self._func_sysd(mt, ut) + self._noise_sysd(mt).random()
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
    
    def _control_max_constraint(self, x, i, max_val):
        _, _, u_i = self._b_u_seg(x, i)
        return max_val - u_i

    def _control_min_constraint(self, x, i, min_val):
        _, _, u_i = self._b_u_seg(x, i)
        return u_i - min_val

    def _mean_final_constraint(self, x, m_des, num_segments):
        m_k, _, _ = self._b_u_seg(x, num_segments-1)
        return m_k - m_des
    
    def _mean_start_constraint(self, x, m_0):
        m_k, _, _ = self._b_u_seg(x, 0)
        return m_k - m_0

    def _belief_constraint(self, x, i, num_segments):
        m_i, s_i, u_i = self._b_u_seg(x, i)
        Cov_i = s_i.reshape(self._dim_state, self._dim_state).transpose()
        m_ip1, Cov_ip1 = self.integrate_belief_segment((m_i, Cov_i), u_i, num_segments)
        s_ip1 = Cov_ip1.transpose().reshape(-1,)
        b_i_vec = np.append(m_i, s_i)
        b_ip1_vec = np.append(m_ip1, s_ip1)
        return b_i_vec - b_ip1_vec

    def integrate_belief_segment(self, b_i, u_i, num_segments):
        """This is to represent equation 18.
             
        phi(b_i', u_i') = F(b_i', u_i') +   sum    F(b_{t+1}, u_i) - F(b_t, u_i)
                                          t in seg

        This essentially returns b_{i+1}'
        """
        m_i, Cov_i = b_i
        b_seg = [self.ekf_update_mlo((m_i, Cov_i), u_i)]
        for t in range(1, self._planning_horizon // num_segments):
            b_seg.append(self.ekf_update_mlo(b_seg[t-1], u_i))
        sum_mean = b_seg[0][0]
        sum_Cov = b_seg[0][1]
        for t in range(len(b_seg)):
            if t + 1 < len(b_seg) - 1:
                sum_mean += (b_seg[t+1][0] - b_seg[t][0])
                sum_Cov += (b_seg[t+1][1] - b_seg[t][1])
        return sum_mean, sum_Cov

    
    def _opt_cost_func_seg(self, x, b_des, u_des, num_segments):
        """This will be used as part of the objective function for scipy
        optimizer. Therefore we only take in one input vector, x.  We require
        the structure of x to be:

        [ m1 | Cov 1 | u1 ] ... [ m_i | Cov i | u_i ]

        Use the _b_u_seg function to obtain the individual parts.
        """
        bu_traj = []
        for i in range(num_segments):
            m_i, s_i, u_i = self._b_u_seg(x, i)
            Cov_i = s_i.reshape(self._dim_state, self._dim_state).transpose()
            bu_traj.append(((m_i, Cov_i), u_i))
        return self.segmented_cost_function(bu_traj, b_des, u_des, num_segments)

    def segmented_cost_function(self, bu_traj, b_des, u_des, num_segments):
        """The cost function in eq 17. 

        Args:
            b_des (tuple): The desired belief (mT, CovT). This is needed to compute the cost function.
            u_des (list): A list of desired controls at the beginning of each segment.
                If this information is not available, pass in an empty list.
        """
        if len(u_des) > 0 and len(u_des) != num_segments:
            raise ValueError("The list of desired controls, if provided, must have one control"\
                             "per segment")

        b_T, u_T = bu_traj[-1]
        m_T, Cov_T = b_T
        s_T = Cov_T.transpose().reshape(-1,)
        
        sLs = np.dot(np.dot(s_T.transpose(), self._L), s_T)
        Summation = 0
        for i in range(num_segments):
            b_i, u_i = bu_traj[i]
            m_i, _ = b_i
            m_i_diff = m_i - b_des[0]
            if len(u_des) > 0:
                u_i_des = u_des[i]
            else:
                u_i_des = np.zeros(self._dim_control)
            u_i_diff = u_i - u_i_des
            Summation += np.dot(np.dot(m_i_diff.transpose(), self._Q), m_i_diff)\
                + np.dot(np.dot(u_i_diff.transpose(), self._R), u_i_diff)
        return sLs + Summation
    
    def create_plan(self, b_0, b_des, u_init, num_segments=5, control_bounds=None,
                    opt_options={}, opt_callback=None):
        """Solves the SQP problem using direct transcription, and produce belief points
        and controls at segments.
        Reference: https://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html"""
        # build initial guess from initial belief and the given trajectory.
        x_0 = []
        b_i = b_0
        for i in range(num_segments):
            m_i, Cov_i = b_i
            s_i = Cov_i.transpose().reshape(-1,)
            u_i = u_init[i]
            x_0.extend(m_i)
            x_0.extend(s_i)
            x_0.extend(u_i)
            b_i = self.integrate_belief_segment(b_i, u_i, num_segments)

        # constraints        
        constraints = []
        ## initial belief constraint
        cons_start = {'type': 'eq',
                      'fun': self._mean_start_constraint,
                      'args': [b_0[0]]}
        constraints.append(cons_start)
        ## belief dynamics constraints and control bounds
        for i in range(num_segments):
            if i + 1 < num_segments-1:
                cons_belief = {'type': 'eq',
                               'fun': self._belief_constraint,
                               'args': [i, num_segments]}
                constraints.append(cons_belief)

                # control bounds
                if control_bounds is not None:
                    cons_control_min = {'type': 'ineq',
                                        'fun': self._control_min_constraint,
                                        'args': [i, control_bounds[0]]}                
                    cons_control_max = {'type': 'ineq',
                                        'fun': self._control_max_constraint,
                                        'args': [i, control_bounds[1]]}
                    constraints.append(cons_control_min)
                    constraints.append(cons_control_max)

        # final belief constraint
        cons_final = {'type': 'eq',
                      'fun': self._mean_final_constraint,
                      'args': [b_des[0], num_segments]}
        constraints.append(cons_final)

        opt_res = optimize.minimize(self._opt_cost_func_seg, x_0,
                                    method="SLSQP",
                                    args=(b_des, u_init, num_segments),
                                    constraints=constraints,
                                    options=opt_options,
                                    callback=opt_callback)
        return opt_res

    def interpret_sqp_plan(self, opt_res, num_segments):
        x_res = opt_res.x
        plan = []
        for i in range(num_segments):
            m_i, s_i, u_i = self._b_u_seg(x_res, i)
            Cov_i = s_i.reshape(self._dim_state, self._dim_state).transpose()
            plan.append((m_i, Cov_i, u_i))
        return plan
