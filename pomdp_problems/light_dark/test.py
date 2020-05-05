import pomdp_problems.light_dark as ld
import pomdp_py
import numpy as np
import time


if __name__ == "__main__":
    env = ld.LightDarkEnvironment(ld.State((0.5, 2.5)),  # init state
                                  (1.5, -1),  # goal pose
                                  5,  # light
                                  1)  # const
    obsmodel = ld.ObservationModel(5, 1)
    func_sysd = env.transition_model.func()
    func_obs = obsmodel.func()
    jac_sysd = env.transition_model.jac_dx()
    jac_sysd_u = env.transition_model.jac_du()
    jac_obs = obsmodel.jac_dx()
    
    num_segments = 5
    L = 200
    Q = np.array([[0.5, 0],
                  [0, 0.5]])
    R = np.array([[0.5, 0],
                  [0, 0.5]])
    blqr = pomdp_py.BLQR(func_sysd, func_obs, jac_sysd, jac_obs, jac_sysd_u,
                         None, L, Q, R)

    b_0 = (np.array([1.3, 2.0]),
          np.array([[0.5, 0.0],
                    [0.0, 0.5]]))
    u_0 = np.array([0.2, 0.2])
    b_des = (np.array([2.2, 0.5]),
             np.array([[0.05, 0.0],
                       [0.0, 0.05]]))
    u_des = [np.array([0.01, 0.01])]*num_segments
    noise = pomdp_py.Gaussian([0,0], [[0.05, 0],
                                      [0, 0.05]])
    print(blqr._create_plan(b_0, u_0, b_des, u_des, noise, num_segments=num_segments))
    # bt = (np.array([1.3, 2.0]), np.array([[0.5, 0.0], [0.0, 0.5]]))
    # ut = np.array([1.5, 0.2])
    
    # func_sysd = env.transition_model.func()
    # func_obs = obsmodel.func()
    # jac_sysd = env.transition_model.jac_dx()
    # jac_sysd_u = env.transition_model.jac_du()
    # jac_obs = obsmodel.jac_dx()
    
    # noise_sysd = pomdp_py.Gaussian([0,0], [[0.1,0],
    #                                        [0,0.1]]).random()
    # noise_obs_cov = obsmodel.noise_covariance(bt[0])

    # blqr = pomdp_py.BLQR(func_sysd, func_obs, jac_sysd, jac_obs, jac_sysd_u,
    #                      None, None, None, None)
    # start = time.time()
    # bnext = blqr.ekf_update_mlo(bt, ut, noise_sysd, noise_obs_cov)
    # print("Time taken:")
    # print(float(time.time() - start))
    # print("Mean:")
    # print(bnext[0])
    # print("Covariance:")
    # print(bnext[1])
