# """
# Defines the belief representation.

# Origin: Belief space planning assuming maximum likelihood observations

# Quote from the paper:

#     Belief state was modeled as an isotropic Gaussian pdf
#     over the state space: :math:`b = (m,s) \in \mathbb{R}^2\times\mathbb{R}^{+}`.

# Technical definition:

# * An **isotropic gaussian** is one where the covariance matrix is
#   represented by the simplified matrix `source <https://math.stackexchange.com/questions/1991961/gaussian-distribution-is-isotropic>`_

# This is an example where you implement a problem-specific
# belief representation, instead of directly using an existing
# one provided by the pomdp_py; The interfaces of pomdp_py does
# support this kind of extension. In this example, the belief
# distribution is a variant of pomdp_py.Gaussian.

# The pomdp_py library does not (currently) include a BeliefState interface,
# since the belief distribution itself is just a Distribution and adding
# such interface makes the library unnecessarily complicated. The existing
# interfaces are fully able to support belief space planning as well.
# """
# import pomdp_py

# class IsotropicGaussian2d(pomdp_py.Gaussian):

#     def __init__(self, mean, sigma):
#         """initialize isotropic Gaussian distribution.
        
#         Args:
#             mean (list): 2d vector of robot position
#             sigma (float): sigma defining the standard deviation of the Gaussian.
#         """
#         if len(mean) != 2:
#             raise ValueError("Mean vector of belief must be of length 2")
#         self._sigma = sigma
#         super().__init__(mean,
#                          [[sigma**2, 0,
#                            0, sigma**2]])
        
#     @property
#     def sigma(self):
#         return self._sigma

    
# class BeliefState(pomdp_py.State):
#     """Belief state;

#     **Note** that the paper makes use of belief-space reward function, R(b,a),
#     which treats a belief distribution as a state, we shall define a
#     BeliefState class that contains an underlying Distribution.  Also, we
#     define belief space dynamics models; See light_dark.models.belief_space.
#     """
#     def __init__(self, belief):
#         if type(belief) != IsotropicGaussian2d:
#             raise TypeError("belief state's belief distribution must be LightDarkBelief")

#         # The belief is LightDarkBelief, which is a Gaussian.
#         self._belief = belief

#         # For hashing, use the mean
#         self._hashcode = hash(tuple(self._belief.mean))

#     @property
#     def belief(self):
#         return self._belief
        
#     def __hash__(self):
#         return self._hashcode
    
#     def __eq__(self, other):
#         if isinstance(other, BeliefState):
#             return self._belief == other.belief
#         else:
#             return False
        
#     def __str__(self):
#         return self.__repr__()
    
#     def __repr__(self):
#         return "BeliefState(%s)" % (str(self._belief.mean, self._belief.sigma))
        
