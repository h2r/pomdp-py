from pomdp_py.framework.basics import BeliefDistribution

class ParticlesBelief(BeliefDistribution):
    def __init__(self, particles):
        self._particles = particles  # each particle is a state
        
    @property
    def particles(self):
        return self._particles

    def __str__(self):
        hist = self.get_histogram()
        hist = [(k,hist[k]) for k in list(reversed(sorted(hist, key=hist.get)))]
        return str(hist)

    def __len__(self):
        return len(self._particles)
    
    def __getitem__(self, state):
        belief = 0
        for s in self._particles:
            if s == state:
                belief += 1
        return belief / len(self._particles)
    
    def __setitem__(self, state, value):
        """Assume that value is between 0 and 1"""
        particles = [s for s in self._particles if s != state]
        len_not_state = len(particles)
        amount_to_add = value * len_not_state / (1 - value)
        for i in range(amount_to_add):
            particles.append(state)
        self._particles = particles
        
    def __hash__(self):
        if len(self._particles) == 0:
            return hash(0)
        else:
            # if the state space is large, a random particle would differentiate enough
            indx = random.randint(0, len(self._particles-1))
            return hash(self._particles[indx])
        
    def __eq__(self, other):
        if not isinstance(other, BeliefDistribution_Particles):
            return False
        else:
            if len(self._particles) != len(other.praticles):
                return False
            state_counts_self = {}
            state_counts_other = {}
            for s in self._particles:
                if s not in state_counts_self:
                    state_counts_self[s] = 0
                state_counts_self[s] += 1
            for s in other.particles:
                if s not in state_counts_self:
                    return False
                if s not in state_counts_other:
                    state_counts_other[s] = 0
                state_counts_other[s] += 1
            return state_counts_self == state_counts_other

    def mpe(self, hist=None):
        mpe_state = None
        if hist is None:
            hist = self.get_histogram()
        for s in hist:
            if mpe_state is None:
                mpe_state = s
            else:
                if hist[s] > hist[mpe_state]:
                    mpe_state = s
        return mpe_state

    def random(self):
        if len(self._particles) > 0:
            return random.choice(self._particles)
        else:
            return None

    def add(self, particle):
        self._particles.append(particle)

    def get_histogram(self):
        state_counts_self = {}
        for s in self._particles:
            if s not in state_counts_self:
                state_counts_self[s] = 0
            state_counts_self[s] += 1
        for s in state_counts_self:
            state_counts_self[s] = state_counts_self[s] / len(self._particles)
        return state_counts_self

    def get_abstraction(self, state_mapper):
        particles = [state_mapper(s) for s in self._particles]
        return particles

