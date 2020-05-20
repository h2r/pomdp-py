from pomdp_py.representations.distribution.weighted_particles cimport WeightedParticles
from pomdp_py.framework.basics cimport State, Action, Observation, TransitionModel, ObservationModel

"""Updating weighted particles"""

cpdef update_weighted_particles_belief(WeightedParticles current_particles,
                                       Action real_action, Observation real_observation,
                                       ObservationModel observation_model,
                                       TransitionModel transition_model,
                                       dict targs={}, dict oargs={}):
    cdef State particle, next_particle
    cdef float weight, next_weight
    cdef float transition_prob
    cdef list new_particles = []
    for particle, _ in current_particles.particles:
        next_particle = transition_model.sample(particle, real_action)

        observation_prob = observation_model.probability(real_observation,
                                                         next_particle,
                                                         real_action,
                                                         **oargs)
        transition_prob = 0
        for particle, weight in current_particles.particles:
            transition_prob +=\
                transition_model.probability(next_particle,
                                             particle,
                                             real_action,
                                             **targs) * weight
        next_weight = observation_prob * transition_prob
        new_particles.append((next_particle, next_weight))
    return WeightedParticles(new_particles)
            
