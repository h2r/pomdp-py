from pomdp_py.framework.basics cimport GenerativeDistribution
import random

cdef class WeightedParticles(GenerativeDistribution):
    """
    Represents a distribution :math:`\Pr(X)` with weighted particles, each is a
    tuple (value, weight). "value" means a value for the random variable X. If
    multiple values are present for the same value, will interpret the
    probability at X=x as the average of those weights.

    __init__(self, list particles, str approx_method="none", distance_func=None)

    Args:
       particles (list): List of (value, weight) tuples. The weight represents
           the likelihood that the value is drawn from the underlying distribution.
       approx_method (str): 'nearest' if when querying the probability
            of a value, and there is no matching particle for it, return
            the probability of the value closest to it. Assuming values
            are comparable; "none" if no approximation, return 0.
       distance_func: Used when approx_method is 'nearest'. Returns
           a number given two values in this particle set.
    """
    def __init__(self, list particles, str approx_method="none", distance_func=None):
        self._values = [value for value, _ in particles]
        self._weights = [weight for _, weight in particles]
        self._particles = particles

        self._hist = self.get_histogram()
        self._hist_valid = True

        self._approx_method = approx_method
        self._distance_func = distance_func

    @property
    def particles(self):
        return self._particles

    @property
    def values(self):
        return self._values

    @property
    def weights(self):
        return self._weights

    def add(self, particle):
        """add(self, particle)
        particle: (value, weight) tuple"""
        self._particles.append(particle)
        s, w = particle
        self._values.append(s)
        self._weights.append(w)
        self._hist_valid = False

    def __str__(self):
        return str(self.condense().particles)

    def __len__(self):
        return len(self._particles)

    def __getitem__(self, value):
        """Returns the probability of `value`; normalized"""
        if len(self.particles) == 0:
            raise ValueError("Particles is empty.")

        if not self._hist_valid:
            self._hist = self.get_histogram()
            self._hist_valid = True

        if value in self._hist:
            return self._hist[value]
        else:
            if self._approx_method == "none":
                return 0.0
            elif self._approx_method == "nearest":
                nearest_dist = float('inf')
                nearest = self._values[0]
                for s in self._values[1:]:
                    dist = self._distance_func(s, nearest)
                    if dist < nearest_dist:
                        nearest_dist = dist
                        nearest = s
                return self[nearest]
            else:
                raise ValueError("Cannot handle approx_method:",
                                 self._approx_method)

    def __setitem__(self, value, prob):
        """
        The particle belief does not support assigning an exact probability to a value.
        """
        raise NotImplementedError

    def random(self):
        """Samples a value based on the particles"""
        value = random.choices(self._values, weights=self._weights, k=1)[0]
        return value

    def mpe(self):
        if not self._hist_valid:
            self._hist = self.get_histogram()
            self._hist_valid = True
        return max(self._hist, key=self._hist.get)

    def __iter__(self):
        return iter(self._particles)

    cpdef dict get_histogram(self):
        """
        get_histogram(self)
        Returns a mapping from value to probability, normalized."""
        cdef dict hist = {}
        cdef dict counts = {}
        # first, sum the weights
        for s, w in self._particles:
            hist[s] = hist.get(s, 0) + w
            counts[s] = counts.get(s, 0) + 1
        # then, average the sums
        total_weights = 0.0
        for s in hist:
            hist[s] = hist[s] / counts[s]
            total_weights += hist[s]
        # finally, normalize
        for s in hist:
            hist[s] /= total_weights
        return hist

    @classmethod
    def from_histogram(cls, histogram):
        """Given a pomdp_py.Histogram return a particle representation of it,
        which is an approximation"""
        particles = []
        for v in histogram:
            particles.append((v, histogram[v]))
        return WeightedParticles(particles)

    def condense(self):
        """
        Returns a new set of weighted particles with unique values
        and weights aggregated (taken average).
        """
        return WeightedParticles.from_histogram(self.get_histogram())


cdef class Particles(WeightedParticles):
    """ Particles is a set of unweighted particles; This set of particles represent
    a distribution :math:`\Pr(X)`. Each particle takes on a specific value of :math:`X`.
    Inherits :py:mod:`~pomdp_py.representations.distribution.particles.WeightedParticles`.

    __init__(self, particles, **kwargs)

    Args:
        particles (list): List of values.
        kwargs: see __init__() of :py:mod:`~pomdp_py.representations.distribution.particles.WeightedParticles`.
    """
    def __init__(self, particles, **kwargs):
        super().__init__(list(zip(particles, [None]*len(particles))), **kwargs)

    def __iter__(self):
        return iter(self.particles)

    def add(self, particle):
        """add(self, particle)
        particle: just a value"""
        self._particles.append((particle, None))
        self._values.append(particle)
        self._weights.append(None)
        self._hist_valid = False

    @property
    def particles(self):
        """For unweighted particles, the particles are just values."""
        return self._values

    def get_abstraction(self, state_mapper):

        """get_abstraction(self, state_mapper)
        feeds all particles through a state abstraction function.
        Or generally, it could be any function.
        """
        particles = [state_mapper(s) for s in self.particles]
        return particles

    @classmethod
    def from_histogram(cls, histogram, num_particles=1000):
        """Given a pomdp_py.Histogram return a particle representation of it,
        which is an approximation"""
        particles = []
        for _ in range(num_particles):
            particles.append(histogram.random())
        return Particles(particles)

    cpdef dict get_histogram(self):
        cdef dict hist = {}
        for s in self.particles:
            hist[s] = hist.get(s, 0) + 1
        for s in hist:
            hist[s] = hist[s] / len(self.particles)
        return hist

    def random(self):
        """Samples a value based on the particles"""
        if len(self._particles) > 0:
            return random.choice(self._values)
        else:
            return None
