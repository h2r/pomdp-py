from pomdp_py import GenerativeDistribution
import sys
import random
import numpy as np

class Histogram(GenerativeDistribution):
    def __init__(self, histogram):
        """`histogram` is a dictionary mapping from variable value to probability"""
        if not (isinstance(histogram, dict)):
            raise ValueError("Unsupported histogram representation! %s" % str(type(histogram)))
        self._histogram = histogram

    @property
    def histogram(self):
        return self._histogram

    def __str__(self):
        if isinstance(self._histogram, dict):
            return str([(k,self._histogram[k])
                        for k in list(reversed(sorted(self._histogram, key=self._histogram.get)))[:5]])

    def __len__(self):
        return len(self._histogram)

    def __getitem__(self, value):
        if value in self._histogram:
            return self._histogram[value]
        else:
            return 0

    def __setitem__(self, value, prob):
        self._histogram[value] = prob

    def __hash__(self):
        if len(self._histogram) == 0:
            return hash(0)
        else:
            # if the domain is large, a random state would differentiate enough
            value = self.random()
            return hash(self._histogram[value])

    def __eq__(self, other):
        if not isinstance(other, Histogram):
            return False
        else:
            for s in self._histogram:
                if s not in other._histogram:
                    return False
                if self[s] != other[s]:
                    return False

    def __iter__(self):
        return iter(self._histogram)

    def mpe(self):
        return max(self._histogram, key=self._histogram.get)

    def random(self):
        """Randomly sample a value based on the probability in the histogram"""
        candidates = list(self._histogram.keys())
        prob_dist = []
        for value in candidates:
            prob_dist.append(self._histogram[value])
        if sys.version_info[0] >= 3 and sys.version_info[1] >= 6:
            # available in Python 3.6+
            return random.choices(candidates, weights=prob_dist, k=1)[0]
        else:
            return np.random.choice(candidates, 1, p=prob_dist)[0]

    def get_histogram(self):
        return self._histogram
    
    # Deprecated; it's assuming non-log probabilities
    def is_normalized(self):
        """Returns true if this distribution is normalized"""
        prob_sum = sum(self._histogram[state] for state in self._histogram)
        return abs(1.0-prob_sum) < EPSILON

