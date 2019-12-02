def abstraction_over_particles(particles, state_mapper):
    particles = [state_mapper(s) for s in particles]
    return particles
