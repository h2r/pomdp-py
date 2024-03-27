from __future__ import annotations
from functools import cached_property
from typing import Iterable, Iterator, Union


cdef class Vector:
    """
    The Vector class. Provides an implementation of a vector for maintaining multiple values.
    """
    cdef list[float] vals
    
    def __init__(self, values: float | int | Iterable[float | int] = list([0.])):
        cdef list _vec = list()
        if isinstance(values, (float, int)):
            _vec.append(values)
        elif isinstance(values, (list, tuple)):
            _vec += list(values)
        else:
            raise TypeError(f"values must be type int, float, list, or tuple, but got {type(values)}.")
        
        # Store the values as an array of floats.
        self.vals = list(float(v) for v in _vec)

    @cached_property
    def values(self) -> list[float]:
        return self.vals.copy()

    def __iter__(self) -> Iterator:
        return iter(self.vals)

    def __len__(self) -> int:
        return len(self.vals)

    def __eq__(self, other: Vector | list) -> bool:
        if not isinstance(other, (Vector, list)):
            raise TypeError(f"other must be type Vector or list, but got {type(other)}.")
        return len(self) == len(other) and all(v0 == v1 for v0, v1 in zip(self, other))

    def __add__(self, other: Vector | list | float | int) -> Vector:
        if isinstance(other, (float, int)):
            vec = [other] * len(self)
        elif isinstance(other, Vector):
            vec = other
        else:
            raise TypeError(f"other must be type Vector, float, or int, but got {type(other)}.")
        return Vector([v0 + v1 for v0, v1 in zip(self, vec)])

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        if not isinstance(other, (float, int)):
            raise TypeError(f"other must be type float or int, but got {type(other)}.")
        return Vector([v * other for v in self])

    def __rmul__(self, other):
        return self.__mul__(other)

    def __str__(self) -> str:
        return str(self.vals)


ResponseVariableType = Union[float, Vector]


cdef class GenericResponse:
    """
    A GenericResponse class maintains variables within a dictionary. However, subclasses of GenericResponse
    can provide access to the dictionary variables using the dot (.) operator. Currently, this class can
    handle arithmetic and comparison operations. However, if special operations will need to be performed, 
    these operations need to be handled in the subclass.
    """
    cdef dict __dict__

    def __init__(self, dict response_dict = dict()):
        self.__dict__.update(response_dict.copy())

    cpdef void _check_reward_compatibility(self, value):
        if not isinstance(value, GenericResponse):
            raise TypeError(f"other must be type GenericResponse, float, or int, but got {type(value)}.")

    cdef dict add_response(self, GenericResponse other):
        self._check_reward_compatibility(other)
        cdef dict rv = dict()
        for name, value in self.__dict__.items():
            rv.update({name: value + other.__dict__[name]})
        return rv

    def __add__(self, other: GenericResponse) -> GenericResponse:
        return GenericResponse(self.add_response(other))

    def __radd__(self, other: GenericResponse) -> GenericResponse:
        return self.__add__(other)

    cpdef dict mul_scalar(self, float other):
        if not isinstance(other, float):
            raise TypeError("other must be type float or int.")
        cdef dict rv = dict()
        for name, value in self.__dict__.items():
            rv.update({name: value * other})
        return rv

    def __mul__(self, other: float | int) -> GenericResponse:
        return GenericResponse(self.mul_scalar(other))

    def __rmul__(self, other) -> GenericResponse:
        return self.__mul__(other)

    def __eq__(self, other: GenericResponse) -> bool:
        self._check_reward_compatibility(other)
        return all(value == other.__dict__[name] for name, value in self.__dict__.items())

    def __ne__(self, other) -> bool:
        self._check_reward_compatibility(other)
        return all(value != other.__dict__[name] for name, value in self.__dict__.items())

    def __lt__(self, other) -> bool:
        self._check_reward_compatibility(other)
        return all(value < other.__dict__[name] for name, value in self.__dict__.items())

    def __le__(self, other) -> bool:
        self._check_reward_compatibility(other)
        return all(value <= other.__dict__[name] for name, value in self.__dict__.items())

    def __gt__(self, other) -> bool:
        self._check_reward_compatibility(other)
        return all(value > other.__dict__[name] for name, value in self.__dict__.items())

    def __ge__(self, other) -> bool:
        self._check_reward_compatibility(other)
        return all(value >= other.__dict__[name] for name, value in self.__dict__.items())
        
    def __str__(self) -> str:
        return ", ".join([f"{name}={values}" for name, values in self.__dict__.items()])


cdef class RewardCost(GenericResponse):

    def __init__(self, float reward=0.0, Vector cost=Vector()):
        super().__init__({"reward": reward, "cost": cost})

    def __add__(self, other: RewardCost) -> RewardCost:
        return RewardCost(**self.add_response(other))

    def __mul__(self, other: float) -> RewardCost:
        return RewardCost(**self.mul_scalar(other))
