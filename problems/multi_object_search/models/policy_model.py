# Policy model for Mos

import pomdp_py
import random
from ..domain.action import *

class PolicyModel(pomdp_py.RolloutPolicy):
    """Simple policy model. All actions are possible at any state."""

    def __init__(self):
        """FindAction can only be taken after LookAction"""
        self._all_actions = set(ALL_ACTIONS)
        self._all_except_detect = self._all_actions - set({FindAction})

    def sample(self, state, **kwargs):
        return random.sample(self._get_all_actions(**kwargs), 1)[0]
    
    def probability(self, action, state, **kwargs):
        raise NotImplemented

    def argmax(self, state, **kwargs):
        """Returns the most likely action"""
        raise NotImplemented

    def get_all_actions(self, state=None, history=None):
        """note: detect can only happen after look."""
        if history is not None and len(history) > 1:
            # last action
            last_action = history[-1][0]
            if isinstance(last_action, LookAction):
                return self._all_actions
        return self._all_except_detect

    def rollout(self, state, history=None):
        return random.sample(self.get_all_actions(history=history), 1)[0]
