from pomdp_py.framework.basics cimport Action, Agent, Observation, State, Option,\
    TransitionModel, ObservationModel, RewardModel

cdef class Planner:
    """A Planner can :meth:`plan` the next action to take for a given agent (online
    planning). The planner can be updated (which may also update the agent belief)
    once an action is executed and observation received.

    A Planner may be a purely planning algorithm, or it could be using a learned
    model for planning underneath the hood. Its job is to output an action to take
    for a given agent.

    You can implement a Planner that is specific to an agent, or not. If specific, then
    when calling :meth:`plan` the agent passed in is expected to always be the same one.
    """
    cpdef public plan(self, Agent agent):
        """
        plan(self, Agent agent)
        The agent carries the information:
        Bt, ht, O,T,R/G, pi, necessary for planning"""
        raise NotImplementedError

    cpdef public update(self, Agent agent, Action real_action, Observation real_observation):
        """
        update(self, Agent agent, Action real_action, Observation real_observation)
        Updates the planner based on real action and observation.
        Updates the agent accordingly if necessary. If the agent's
        belief is also updated here, the `update_agent_belief`
        attribute should be set to True. By default, does nothing."""
        pass

    def updates_agent_belief(self):
        """True if planner's update function also updates agent's
        belief."""
        return False
