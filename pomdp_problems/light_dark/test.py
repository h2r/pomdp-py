import pomdp_problems.light_dark as ld
import matplotlib.pyplot as plt
import pomdp_py
import numpy as np
import time

count = 0

# true location.
x_0 = np.array([2.5, 0.0])
goal_pos = np.array([0.0, 0.0])

# defines the observation noise equation.
light = 5
const = 1

# planning horizon
planning_horizon = 30

# number of segments for direct transcription
num_segments = 10

var_sysd = 1e-9

# Environment.
env = ld.LightDarkEnvironment(x_0,  # init state
                              light,  # light
                              const)  # const
obsmodel = ld.ObservationModel(light, const)

func_sysd = env.transition_model.func()
func_obs = obsmodel.func()
jac_sysd = env.transition_model.jac_dx()
jac_sysd_u = env.transition_model.jac_du()
jac_obs = obsmodel.jac_dx()
noise_obs = obsmodel.func_noise()
noise_sysd = env.transition_model.func_noise(var_sysd)

L = 200
Q = np.array([[0.5, 0],
              [0, 0.5]])
R = np.array([[0.5, 0],
              [0, 0.5]])

b_des = (goal_pos,
         np.array([[1e-6, 0.0],
                   [0.0, 1e-6]]))
# u_des = [0.5*np.random.rand(2) for _ in range(num_segments)]
u_des = [[0.0, 0.0] for _ in range(num_segments)]

blqr = pomdp_py.BLQR(func_sysd, func_obs, jac_sysd, jac_obs, jac_sysd_u,
                     noise_obs, noise_sysd,
                     None, L, Q, R,
                     planning_horizon=planning_horizon)


def manual_test(blqr):
    # Initial and final belief states.
    b_0 = (np.array([2.0, 2.0]),
           np.array([[5.0, 0.0],
                     [0.0, 5.0]]))
    print("Path through the light:")
    print(b_0[0])
    print(b_0[1])
    with np.printoptions(precision=3, suppress=True):
        b_1 = blqr.ekf_update_mlo(b_0, [1.0, 0.0])
        print(b_1[0])
        print(b_1[1])
        b_1 = blqr.ekf_update_mlo(b_1, [1.0, 0.0])
        print(b_1[0])
        print(b_1[1])
        b_1 = blqr.ekf_update_mlo(b_1, [0.0, -1.0])
        print(b_1[0])
        print(b_1[1])
        b_1 = blqr.ekf_update_mlo(b_1, [0.0, -1.0])
        print(b_1[0])
        print(b_1[1])
        b_1 = blqr.ekf_update_mlo(b_1, [-1.0, 0.0])
        print(b_1[0])
        print(b_1[1])    
        b_1 = blqr.ekf_update_mlo(b_1, [-1.0, 0.0])
        print(b_1[0])
        print(b_1[1])
        b_1 = blqr.ekf_update_mlo(b_1, [-1.0, 0.0])
        print(b_1[0])
        print(b_1[1])
        b_1 = blqr.ekf_update_mlo(b_1, [-1.0, 0.0])
        print(b_1[0])
        print(b_1[1])
        b_1 = blqr.ekf_update_mlo(b_1, [-0.99, 0.09])
        print(b_1[0])
        print(b_1[1])


    # Initial and final belief states.
    b_0 = (np.array([2.0, 2.0]),
           np.array([[5.0, 0.0],
                     [0.0, 5.0]]))
    print("Path directly to goal")
    print(b_0[0])
    print(b_0[1])
    with np.printoptions(precision=3, suppress=True):
        b_1 = blqr.ekf_update_mlo(b_0, [-1.0, -1.0])
        print(b_1[0])
        print(b_1[1])
        b_1 = blqr.ekf_update_mlo(b_1, [-0.99, -0.99])
        print(b_1[0])
        print(b_1[1])

    ####### FOR DEBUGGING
    # traj through light
    u_traj_light = [[1.0, 0.0],
                    [1.0, 0.0],
                    [0.0, -1.0],
                    [0.0, -1.0],
                    [-1.0, 0.0],
                    [-1.0, 0.0],
                    [-1.0, 0.0],
                    [-1.0, 0.0],
                    [-0.99, 0.09]]
    b_t = b_0
    b_traj_light = [b_t]
    for t in range(len(u_traj_light)):
        u_t = u_traj_light[t]
        b_tp1 = blqr.ekf_update_mlo(b_t, u_t)
        b_traj_light.append(b_tp1)
        b_t = b_tp1
    bu_traj_light = [(b_traj_light[t], np.array(u_traj_light[t])) for t in range(len(u_traj_light))]
    
    # traj not through light
    u_traj_dark = [[-1., -1.], [-1., -1.], [-1., -1.]]
    b_t = b_0
    b_traj_dark = [b_t]
    for t in range(len(u_traj_dark)):
        u_t = u_traj_dark[t]
        b_tp1 = blqr.ekf_update_mlo(b_t, u_t)
        b_traj_dark.append(b_tp1)
        b_t = b_tp1        
    bu_traj_dark = [(b_traj_dark[t], np.array(u_traj_dark[t])) for t in range(len(u_traj_dark))]


    total_light = 0
    total_dark = 0    
    for i in range(1000):
        cost_light = blqr.segmented_cost_function(bu_traj_light, b_des, [], len(bu_traj_light))
        cost_dark = blqr.segmented_cost_function(bu_traj_dark, b_des, [], len(bu_traj_dark))
        total_light += cost_light
        total_dark += cost_dark
    print("avg cost light: %.3f" % (total_light/1000.0))
    print("avg cost dark: %.3f" % (total_dark/1000.0))

    x_range = (-1, 7)
    y_range = (-2, 4)
    viz = ld.LightDarkViz(env, x_range, y_range, 0.1)
    viz.set_goal(goal_pos)
    viz.set_initial_belief_pos(b_0[0])
    
    # Made up paths
    viz.log_position(tuple(b_0[0]), path=2)
    viz.log_position(tuple(b_0[0]), path=3)
    sysd_b_light = [b_0]
    for b_t, u_t in bu_traj_light:
        viz.log_position(tuple(b_t[0]), path=2)
        viz.log_position(tuple(sysd_b_light[-1][0]), path=3)
        sysd_b_light.append((func_sysd(sysd_b_light[-1][0], u_t), 0))

    # Made up paths
    viz.log_position(tuple(b_0[0]), path=4)
    viz.log_position(tuple(b_0[0]), path=5)
    sysd_b_dark = [b_0]
    for b_t, u_t in bu_traj_dark:
        viz.log_position(tuple(b_t[0]), path=4)
        viz.log_position(tuple(sysd_b_dark[-1][0]), path=5)
        sysd_b_dark.append((func_sysd(sysd_b_dark[-1][0], u_t), 0))        

    viz.plot(path_colors={2: [(0,0,0), (0,255,0)],
                          3: [(0,0,0), (0,255,255)],
                          4: [(0,0,0), (255,255,0)],
                          5: [(0,0,0), (255,0,255)]},
             path_styles={2: "--",
                          3: "-",
                          4: "--",
                          5: "-"})
    plt.show()    



def opt_callback(xk, *args):
    global count
    print("Iteration %d" % count)
    with np.printoptions(precision=3, suppress=True):
        print(xk)
    count += 1


def test(blqr):
    ############
    b_0 = (np.array([2.0, 2.0]),
           np.array([[5.0, 0.0],
                     [0.0, 5.0]]))    
    x_sol = blqr.create_plan(b_0, b_des, u_des,
                             num_segments=num_segments,
                             opt_options={"maxiter": 30},
                             opt_callback=opt_callback,
                             control_bounds=(-0.1, 0.1))
    with np.printoptions(precision=3, suppress=True):    
        print("SLSQP solution:")
        print(x_sol)
    plan = blqr.interpret_sqp_plan(x_sol, num_segments)

    # Visualization
    x_range = (-1, 7)
    y_range = (-2, 4)
    viz = ld.LightDarkViz(env, x_range, y_range, 0.1)
    viz.set_goal(goal_pos)
    viz.set_initial_belief_pos(b_0[0])
    viz.log_position(tuple(b_0[0]), path=0)
    viz.log_position(tuple(b_0[0]), path=1)

    sysd_b_plan = [b_0]
    for m_i, _, _ in plan:
        viz.log_position(tuple(m_i), path=0)

    viz.plot(path_colors={0: [(0,0,0), (0,255,0)],
                          1: [(0,0,0), (255,0,0)]},
             path_styles={0: "--",
                          1: "-"},
             path_widths={0: 4,
                          1: 1})
    plt.show()    

    


if __name__ == "__main__":
    test(blqr)
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
