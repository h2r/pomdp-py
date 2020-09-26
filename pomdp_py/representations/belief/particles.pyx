from pomdp_py.representations.distribution.particles cimport Particles
from pomdp_py.framework.basics cimport State, Action, Observation, Agent,\
    TransitionModel, ObservationModel, BlackboxModel
import copy

"""To update particle belief, there are two possibilities.
Either, an algorithm such as POMCP is used which will update
the belief automatically when the planner gets updated; Or,
a standalone particle belief distribution is updated. In
the latter case, the belief update algorithm is as described
in :cite:`silver2010monte` but written explicitly instead
of done implicitly when the MCTS tree truncates as in POMCP.
"""

def abstraction_over_particles(particles, state_mapper):
    particles = [state_mapper(s) for s in particles]
    return particles

cpdef particle_reinvigoration(Particles particles,
                              int num_particles, state_transform_func=None):
    """Note that particles should contain states that have already made
    the transition as a result of the real action. Therefore, they simply
    form part of the reinvigorated particles. At least maintain `num_particles`
    number of particles. If already have more, then it's ok.
    """
    # If not enough particles, introduce artificial noise to existing particles (reinvigoration)
    cdef Particles new_particles = copy.deepcopy(particles)
    if len(new_particles) == 0:
        raise ValueError("Particle deprivation.")

    if len(new_particles) > num_particles:
        return new_particles
    
    print("Particle reinvigoration for %d particles" % (num_particles - len(new_particles)))
    cdef State next_state    
    while len(new_particles) < num_particles:
        # need to make a copy otherwise the transform affects states in 'particles'
        next_state = copy.deepcopy(particles.random())
        # Add artificial noise
        if state_transform_func is not None:
            next_state = state_transform_func(next_state)
        new_particles.add(next_state)
    return new_particles


cpdef update_particles_belief(Particles current_particles,
                              Action real_action, Observation real_observation=None,
                              ObservationModel observation_model=None,
                              TransitionModel transition_model=None,
                              BlackboxModel blackbox_model=None,
                              state_transform_func=None):
    """
    update_particles_belief(Particles current_particles,
                           Action real_action, Observation real_observation=None,
                           ObservationModel observation_model=None,
                           TransitionModel transition_model=None,
                           BlackboxModel blackbox_model=None,
                           state_transform_func=None)
    This is the second case (update particles belief explicitly); Either
    BlackboxModel is not None, or TransitionModel and ObservationModel are not
    None. Note that you DON'T need to call this function if you are using POMCP.
    |TODO: not tested|

    Args:
        state_transform_func (State->State) is used to add artificial noise to
            the reinvigorated particles.
    """
    cdef State particle, next_state
    cdef Observation observation
    cdef list filtered_particles = []
    for particle in current_particles.particles:
        # particle represents a state
        if blackbox_model is not None:
            # We're using a blackbox generator; (s',o,r) ~ G(s,a)
            result = blackbox_model.sample(particle, real_action)
            next_state = result[0]
            observation = result[1]
        else:
            # We're using explicit models
            next_state = transition_model.sample(particle, real_action)
            observation = observation_model.sample(next_state, real_action)
        # If observation matches real, then the next_state is accepted
        if observation == real_observation:
            filtered_particles.append(next_state)
    # Particle reinvigoration
    return particle_reinvigoration(Particles(filtered_particles), len(current_particles.particles),
                                   state_transform_func=state_transform_func)
    
