from pomdp_py.representations.distribution.particles cimport Particles
from pomdp_py.framework.basics cimport State
import copy

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

