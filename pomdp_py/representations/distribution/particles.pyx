from pomdp_py.framework.basics cimport GenerativeDistribution
import random

cdef class Particles(GenerativeDistribution):
    #TODO: inherit WeightedParticles
    def __init__(self, particles, _hash_seed=100):
        self._particles = particles  # each particle is a value
        if len(particles) > 0:
            self._rnd_hash_idx = random.Random(_hash_seed)\
                                       .randint(0, len(particles)-1)

    @property
    def particles(self):
        return self._particles

    def __str__(self):
        hist = self.get_histogram()
        hist = [(k,hist[k]) for k in list(reversed(sorted(hist, key=hist.get)))]
        return str(hist)

    def __len__(self):
        return len(self._particles)

    def __getitem__(self, value):
        """__getitem__(self, value)
        Returns the probability of `value`."""
        belief = 0
        for s in self._particles:
            if s == value:
                belief += 1
        return belief / len(self._particles)

    def __setitem__(self, value, prob):
        """__setitem__(self, value, prob)
        The particle representation is assumed to be not mutable"""
        raise NotImplementedError

    def __hash__(self):
        if len(self._particles) == 0:
            return hash(0)
        else:
            # if the value space is large, a random particle would differentiate enough
            return hash(self._particles[self._rnd_hash_idx])

    def __eq__(self, other):
        if not isinstance(other, Particles):
            return False
        else:
            if len(self._particles) != len(other.praticles):
                return False
            hist = self.get_histogram()
            other_hist = other.get_histogram()
            return hist == other_hist

    def mpe(self, hist=None):
        """
        mpe(self, hist=None)
        Choose a particle that is most likely to be sampled.
        """
        if hist is None:
            hist = self.get_histogram()
        return max(hist, key=hist.get)

    def random(self):
        """random(self)
        Randomly choose a particle"""
        if len(self._particles) > 0:
            return random.choice(self._particles)
        else:
            return None

    def add(self, particle):
        """add(self, particle)
        Add a particle."""
        self._particles.append(particle)

    def get_histogram(self):
        """get_histogram(self)
        Returns a dictionary from value to probability of the histogram"""
        value_counts_self = {}
        for s in self._particles:
            if s not in value_counts_self:
                value_counts_self[s] = 0
            value_counts_self[s] += 1
        for s in value_counts_self:
            value_counts_self[s] = value_counts_self[s] / len(self._particles)
        return value_counts_self

    def get_abstraction(self, state_mapper):
        """get_abstraction(self, state_mapper)
        feeds all particles through a state abstraction function.
        Or generally, it could be any function.
        """
        particles = [state_mapper(s) for s in self._particles]
        return particles

    @classmethod
    def from_histogram(self, histogram, num_particles=1000):
        """from_histogram(self, histogram, num_particles=1000)
        Given a Histogram distribution `histogram`, return
        a particle representation of it, which is an approximation.
        """
        particles = []
        for i in range(num_particles):
            particles.append(histogram.random())
        return Particles(particles)
