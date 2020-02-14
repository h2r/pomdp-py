from pomdp_py.representations.distribution.particles cimport Particles
from pomdp_py.framework.basics cimport State, Action, Observation, Agent,\
    TransitionModel, ObservationModel, BlackboxModel

cpdef particle_reinvigoration(Particles particles, int num_particles, state_transform_func=*)
cpdef update_particles_belief(Particles current_particles, Action real_action, Observation real_observation=*, ObservationModel observation_model=*, TransitionModel transition_model=*, BlackboxModel blackbox_model=*, state_transform_func=*)
