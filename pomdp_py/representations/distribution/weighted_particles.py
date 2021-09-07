from pomdp_py.framework.basics import GenerativeDistribution
import random

class WeightedParticles(GenerativeDistribution):
    def __init__(self, particles, _hash_seed=100):
        """
        Represents a distribution Pr(X) with weighted particles
        'value' means a value for the random variable X. If multiple
        values are present for the same value, will interpret the
        probability at X=x as the average of those weights.

        Args:
            particles (list): List of (value, weight) tuples.
                The weight represents the likelihood that the
                value is drawn from the underlying distribution.
        """
        self._particles = particles
        self._rnd_hash_idx = random.Random(_hash_seed).randint(0, len(particles)-1)
        self._hist = self.get_histogram()

    @property
    def particles(self):
        return self._particles

    @property
    def values(self):
        return [value for value, _ in self._particles]

    @property
    def weights(self):
        return [weight for _, weight in self._particles]

    def __str__(self):
        hist = [(k, self._hist[k]) for k in list(reversed(sorted(self._hist, key=hist.get)))]
        return str(hist)

    def __len__(self):
        return len(self._particles)

    def __getitem__(self, value):
        """Returns the probability of `value`"""
        return self._hist[value]

    def __setitem__(self, value, prob):
        """
        The particle representation is assumed to be not mutable
        """
        raise NotImplementedError

    def __contains__(self, value):
        return value in self._hist

    def __hash__(self):
        if len(self._particles) == 0:
            return hash(0)
        else:
            # if the value space is large, a random particle would differentiate enough
            return hash(self._particles[self._rnd_hash_idx])

    def __eq__(self, other):
        if not isinstance(other, WeightedParticles):
            return False
        else:
            hist = self._hist
            other_hist = other._hist
            return hist == other_hist

    def random(self):
        """Samples a value based on the particles"""
        value, _ = random.choices(self._particles, weights=self.weights, k=1)[0]
        return value

    def mpe(self, hist=None):
        if hist is None:
            hist = self.get_histogram()
        return max(hist, key=hist.get)

    def __iter__(self):
        return iter(self._particles)

    def get_histogram(self):
        hist = {}
        counts = {}
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
    def from_histogram(self, histogram):
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
