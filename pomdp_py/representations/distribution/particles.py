from pomdp_py import GenerativeDistribution

class Particles(GenerativeDistribution):
    def __init__(self, particles):
        self._particles = particles  # each particle is a value
        
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
        belief = 0
        for s in self._particles:
            if s == value:
                belief += 1
        return belief / len(self._particles)
    
    def __setitem__(self, value, prob):
        """Assume that value is between 0 and 1"""
        particles = [s for s in self._particles if s != value]
        len_not_value = len(particles)
        amount_to_add = prob * len_not_value / (1 - prob)
        for i in range(amount_to_add):
            particles.append(value)
        self._particles = particles
        
    def __hash__(self):
        if len(self._particles) == 0:
            return hash(0)
        else:
            # if the value space is large, a random particle would differentiate enough
            indx = random.randint(0, len(self._particles-1))
            return hash(self._particles[indx])
        
    def __eq__(self, other):
        if not isinstance(other, BeliefDistribution_Particles):
            return False
        else:
            if len(self._particles) != len(other.praticles):
                return False
            value_counts_self = {}
            value_counts_other = {}
            for s in self._particles:
                if s not in value_counts_self:
                    value_counts_self[s] = 0
                value_counts_self[s] += 1
            for s in other.particles:
                if s not in value_counts_self:
                    return False
                if s not in value_counts_other:
                    value_counts_other[s] = 0
                value_counts_other[s] += 1
            return value_counts_self == value_counts_other

    def mpe(self, hist=None):
        mpe_value = None
        if hist is None:
            hist = self.get_histogram()
        for s in hist:
            if mpe_value is None:
                mpe_value = s
            else:
                if hist[s] > hist[mpe_value]:
                    mpe_value = s
        return mpe_value

    def random(self):
        if len(self._particles) > 0:
            return random.choice(self._particles)
        else:
            return None

    def add(self, particle):
        self._particles.append(particle)

    def get_histogram(self):
        value_counts_self = {}
        for s in self._particles:
            if s not in value_counts_self:
                value_counts_self[s] = 0
            value_counts_self[s] += 1
        for s in value_counts_self:
            value_counts_self[s] = value_counts_self[s] / len(self._particles)
        return value_counts_self

    def get_abstraction(self, state_mapper):
        particles = [state_mapper(s) for s in self._particles]
        return particles

