from abc import ABC, abstractmethod

class Planner(ABC):
    ''' Abstract class for a Planner. '''

    def __init__(self, mdp, name="planner", init_default_fields=True):

        self.name = name

    # DEPRECATED!
    def plan_and_execute_next_action(self):
        """This function is supposed to plan an action, execute that action,
        and update the belief"""
        raise NotImplemented
    
    # DEPRECATED!
    def execute_next_action(self, action):
        """Execute the given action, and update the belief; Useful
        for cases where the user provide's the action for debugging."""
        raise NotImplemented

    # There is no point for the planner to care about how to execute next action.
    # It should only care about what changes it needs to make after an action
    # has been executed.
    @abstractmethod
    def plan_next_action(self):
        pass
    
    @abstractmethod
    def update(self, real_action, real_observation):
        """update the planner (e.g. truncate the search tree) after
        a real action has been taken and a real observation obtained"""
        pass

    @property
    @abstractmethod
    def params(self):
        """returns a dictionary of parameter names to values"""
        pass
