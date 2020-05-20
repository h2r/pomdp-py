from pomdp_py.framework.basics cimport GenerativeDistribution
import sys
import numpy as np
import random

cdef class WeightedParticles(GenerativeDistribution):
    """Each particle represents a state, and a probability of
    it being sampled from the underlying distribution.
    A WeightedParticles object is immutable."""
    def __init__(self, particles):
        """def __init__(self, particles)
        Args:
            particles (list): a list of particles. Each
                particle is a tuple (value, weight). The weights
                are not required to be normalized.
        """
        self._particles = particles
        # For more efficient sampling
        self._hist = self._get_histogram()
        self._values = [s[0] for s in particles]
        self._weights = [s[1] for s in particles]
    
    property particles:
        def __get__(self):
            """particles getter"""
            return self._particles

    property histogram:
        def __get__(self):
            """histogram getter"""
            return self._hist

    def __getitem__(WeightedParticles self, value):
        """__getitem__(self, value)
        Returns the probability of `value`."""
        cdef float belief = 0
        cdef float tot_weight = 0
        cdef float w
        for s, w in self._particles:
            if s == value:
                belief += w
            tot_weight += w
        return belief / tot_weight

    def __setitem__(WeightedParticles self, value, float prob):
        """This is not supported"""
        raise NotImplementedError("Weighted particles is not mutable.")

    # Not implemented: __hash__, __eq__

    cpdef _get_histogram(WeightedParticles self):
        cdef dict probs = {}
        cdef float w
        cdef tot_weight = 0
        for s, w in self._particles:
            if s not in probs:
                probs[s] = 0.0
            probs[s] += w
            tot_weight += w
        for s in probs:
            probs[s] = probs[s] / tot_weight
        return probs
            
    cpdef mpe(WeightedParticles self):
        """
        mpe(self, hist=None)
        Choose a particle that is most likely to be sampled.
        """
        return max(self._hist, key=self._hist.get)

    cpdef random(WeightedParticles self):
        """
        random(self)
        Choose a particle at random
        """
        # Requires python > 3.7
        if sys.version_info[0] >= 3 and sys.version_info[1] >= 6:
            return random.choices(self._values, weights=self._weights, k=1)[0]
        else:
            idx = np.random.choice(np.arange(len(self._values)), 1, p=self._weights)[0]
            return self._values[idx]
