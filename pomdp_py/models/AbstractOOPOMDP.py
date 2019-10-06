from pomdp_py import State, OOPOMDP, AbstractPOMDP

class AbstractOOPOMDP(OOPOMDP, AbstractPOMDP):

    SEARCH = AbstractPOMDP.SEARCH
    BACKTRACK = AbstractPOMDP.BACKTRACK

    def __init__(self, attributes, domains,
                 abstract_actions,
                 abstract_transition_func,
                 abstract_reward_func,
                 abstract_obserfvation_func,
                 init_abstract_belief,
                 init_abstract_state=None, # this is deprecated. See POMDP
                 gamma=0.99):
        OOPOMDP.__init__(self,
                         attributes,
                         domains,
                         abstract_actions,
                         abstract_transition_func,
                         abstract_reward_func,
                         abstract_obserfvation_func,
                         init_abstract_belief,
                         init_objects_state=init_abstract_state,
                         gamma=gamma)

        
