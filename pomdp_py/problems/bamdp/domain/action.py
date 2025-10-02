import pomdp_py 

class Action(pomdp_py.Action):
    """Action for BAMDP, which includes the name of the action (e.g., the edge to traverse).
    
    Attributes:
        name (str): the name of the action.
    """

    def __init__(self, name):
        self.name = name

    def __hash__(self):
        return hash(self.name)
    
    def __eq__(self, other):
        return self.name == other.name
    
A1 = Action("A1")
A2 = Action("A2")

ACTIONS = [A1, A2]