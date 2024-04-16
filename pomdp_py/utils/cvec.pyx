# cython: profile=True

from __future__ import annotations
cimport cython
from libc.math cimport fmin, fmax
from typing import Iterator


cdef vectord_t null_vector(unsigned int n_zeros) except *:
    cdef vectord_t vec
    vec.assign(n_zeros, 0.0)
    return vec


@cython.boundscheck(False)
cpdef vectord_t list_to_vectord(list[float] values):
    cdef int length = len(values)
    cdef unsigned int i = 0
    cdef vectord_t rv = vectord_t(length)

    if length > 0:
        for i in range(length):
            rv[i] = <double> values[i]
    return rv


@cython.boundscheck(False)
cpdef list[float] vectord_to_list(vectord_t values):
    cdef int length = len(values)
    cdef unsigned int i = 0
    cdef list[float] rv = list()

    if length > 0:
        for i in range(length):
            rv.append(float(values[i]))
    return rv


@cython.boundscheck(False)
cdef double vector_dot_prod(const vectord_t& v0, const vectord_t& v1) except *:
    if v0.size() != v1.size():
        raise ValueError("Both vectors must have the same size.")
    if v0.size() == 0:
        raise ValueError("Vectors should contain at least one value.")

    cdef unsigned int i = 0
    cdef double res = 0.0
    for i in range(v0.size()):
        res += (v0[i] * v1[i])
    return res


@cython.boundscheck(False)
cdef void vector_add(const vectord_t& v0, const vectord_t& v1, vectord_t& res) except *:
    if v0.size() != v1.size():
        raise ValueError("Both vectors must have the same size.")
    if v0.size() == 0:
        raise ValueError("Vectors should contain at least one value.")

    res = vectord_t(v0.size())
    cdef unsigned int i = 0
    for i in range(v0.size()):
        res[i] = v0[i] + v1[i]


@cython.boundscheck(False)
cdef void vector_adds(const vectord_t& v, const double& scalar, vectord_t& res) except *:
    if v.size() == 0:
        raise ValueError("Vector should contain at least one value.")

    res = vectord_t(v.size())
    cdef unsigned int i = 0
    for i in range(v.size()):
        res[i] = v[i] + scalar


@cython.boundscheck(False)
cdef void vector_muls(const vectord_t& v, const double& scalar, vectord_t& res) except *:
    cdef int n_values = v.size()
    if n_values == 0:
        raise ValueError("Vector should contain at least one value.")

    res = vectord_t(n_values)
    cdef unsigned int i = 0
    for i in range(n_values):
        res[i] = v[i] * scalar


@cython.boundscheck(False)
cdef void vector_sub(const vectord_t& v0, const vectord_t& v1, vectord_t& res) except *:
    if v0.size() != v1.size():
        raise ValueError("Both vectors must have the same size.")
    if v0.size() == 0:
        raise ValueError("Vectors should contain at least one value.")

    res = vectord_t(v0.size())
    cdef unsigned int i = 0
    for i in range(v0.size()):
        res[i] = v0[i] - v1[i]


@cython.boundscheck(False)
cdef void vector_subvs(const vectord_t& v, const double& scalar, vectord_t& res) except *:
    cdef int n_values = v.size()
    if n_values == 0:
        raise ValueError("Vector should contain at least one value.")

    res = vectord_t(n_values)
    cdef unsigned int i = 0
    for i in range(n_values):
        res[i] = v[i] - scalar


@cython.boundscheck(False)
cdef void vector_subsv(const double& scalar, const vectord_t& v, vectord_t& res) except *:
    cdef int n_values = v.size()
    if n_values == 0:
        raise ValueError("Vector should contain at least one value.")

    res = vectord_t(n_values)
    cdef unsigned int i = 0
    for i in range(n_values):
        res[i] = scalar - v[i]


@cython.boundscheck(False)
cdef void vector_scalar_div(const vectord_t& v, const double& scalar, vectord_t& res) except *:
    cdef int n_values = v.size()
    if n_values == 0:
        raise ValueError("Vector should contain at least one value.")
    if scalar == 0.0:
        raise ZeroDivisionError("Scalar division by zero!")

    res = vectord_t(n_values)
    cdef unsigned int i = 0
    for i in range(n_values):
        res[i] = v[i] / scalar


@cython.boundscheck(False)
cdef double vector_max(const vectord_t& v) except *:
    cdef int n_values = v.size()
    if n_values == 0:
        raise ValueError("Vector should contain at least one value.")
    if n_values == 1:
        return v[0]

    cdef double max_value = v[0]
    cdef int i = 0
    for i in range(1, n_values):
        if v[i] > max_value:
            max_value = v[i]
    return max_value


@cython.boundscheck(False)
cdef double vector_min(const vectord_t& v) except *:
    cdef int n_values = v.size()
    if n_values == 0:
        raise ValueError("Vector should contain at least one value.")
    if n_values == 1:
        return v[0]

    cdef double min_value = v[0]
    cdef int i = 0
    for i in range(1, n_values):
        if v[i] < min_value:
            min_value = v[i]
    return min_value


@cython.boundscheck(False)
cdef void vector_clip(vectord_t& v, const double& min_value, const double& max_value) except *:
    cdef int n_values = v.size()
    if n_values == 0:
        raise ValueError("Vector should contain at least one value.")
    if min_value >= max_value:
        raise ValueError(
            f"Min value ({min_value}) must be less than max value ({max_value})."
        )
    cdef int i = 0
    for i in range(n_values):
        v[i] = fmax(min_value, fmin(max_value, v[i]))


@cython.boundscheck(False)
cdef void vector_copy(const vectord_t& src, vectord_t& dst) except *:
    cdef int n_values = src.size()
    if n_values == 0:
        raise ValueError("Vector should contain at least one value.")
    dst = vectord_t(n_values)
    cdef int i = 0
    for i in range(n_values):
        dst[i] = src[i]


cdef class Vector:
    """
    The Vector class. Provides an implementation of a vector for
    maintaining multiple values.
    """

    def __init__(self, values: list | tuple):
        if not isinstance(values, (list, tuple)):
            raise TypeError(f"Unhandled type: {type(values)}.")
        if len(values) == 0:
            raise ValueError("The length of values must have at least one value.")
        if not all(isinstance(v, (float, int)) for v in values):
            raise ValueError("All values must be type float or int.")

        self._vals = list_to_vectord(values)
        self._length = self._vals.size()

    cdef bint _is_in_range(Vector self, int index):
        return 0 <= index < self._length

    def as_list(self) -> list[float]:
        """
        Returns a list of the internal values.
        """
        return vectord_to_list(self._vals)

    def as_vector(self) -> vectord_t:
        cdef vectord_t copy
        vector_copy(self._vals, copy)
        return copy

    @staticmethod
    def clip(vec: Vector, min_value: float, max_value: float) -> Vector:
        """
        Clips the values within the value using the given min and max values.
        """
        if not isinstance(vec, Vector):
            raise TypeError("vec must be a Vector.")
        cdef vectord_t rv = vec.as_vector()
        vector_clip(rv, min_value, max_value)
        return Vector(vectord_to_list(rv))

    cpdef Vector copy(Vector self):
        """
        Returns a copy of this vector.
        """
        return Vector(self.as_list())

    cpdef double dot(Vector self, Vector other):
        """
        Performs the dot product between two Vectors.
        """
        if not isinstance(other, Vector):
            raise TypeError("other must be type Vector.")
        return vector_dot_prod(self._vals, other._vals)

    @staticmethod
    def fill(value: float, n_values: int) -> Vector:
        return Vector([value] * n_values)

    cpdef int len(Vector self):
        return self._length

    cdef double max(Vector self):
        return vector_max(self._vals)

    cdef double min(Vector self):
        return vector_min(self._vals)

    @staticmethod
    def null(n_zeros: int) -> Vector:
        return Vector.fill(0.0, n_zeros)

    def __getitem__(self, index: int) -> float:
        index = int(index)
        if not self._is_in_range(index):
            raise IndexError(f"index is out-of-range.")
        return self._vals[index]

    def __setitem__(self, index: int, value: float) -> None:
        index = int(index)
        if not self._is_in_range(index):
            raise IndexError(f"index is out-of-range.")
        if not isinstance(value, float):
            raise TypeError(f"value must be type float, but got type {type(value)}.")
        self._vals[index] = <double> value

    def __iter__(self) -> Iterator:
        return iter(self._vals)

    def __len__(self) -> int:
        return self._length

    def __eq__(self, other: Vector | list | tuple) -> bool:
        if not isinstance(other, (Vector, list, tuple)):
            raise TypeError(
                f"other must be type Vector, list, or tuple, but got {type(other)}."
            )
        if self._length != len(other):
            return False
        return all(v0 == v1 for v0, v1 in zip(self, other))

    def __add__(self, other: Vector | float | int) -> Vector:
        if isinstance(other, (float, int)):
            vector_adds(self._vals, other, self._res_buff)
        elif isinstance(other, Vector):
            vector_add(self._vals, other.as_vector(), self._res_buff)
        else:
            raise TypeError(
                "other must be type Vector with the same length, "
                f"float, or int, but got {type(other)}."
            )
        return Vector(self._res_buff)

    def __radd__(self, other: Vector | float | int) -> Vector:
        return self.__add__(other)

    def __mul__(self, other: float | int) -> Vector:
        if not isinstance(other, (float, int)):
            raise TypeError(f"other must be type float or int, but got {type(other)}.")
        vector_muls(self._vals, other, self._res_buff)
        return Vector(self._res_buff)

    def __rmul__(self, other: float | int) -> Vector:
        return self.__mul__(other)

    def __sub__(self, other: Vector | float | int) -> Vector:
        if isinstance(other, (float, int)):
            vector_subvs(self._vals, other, self._res_buff)
        elif isinstance(other, Vector):
            vector_sub(self._vals, other.as_vector(), self._res_buff)
        else:
            raise TypeError(
                "other must be type Vector with the same length, "
                f"float, or int, but got {type(other)}."
            )
        return Vector(self._res_buff)

    def __rsub__(self, other: Vector | float | int) -> Vector:
        if isinstance(other, (float, int)):
            vector_subsv(other, self._vals, self._res_buff)
        elif isinstance(other, Vector):
            vector_sub(other.as_vector(), self._vals, self._res_buff)
        else:
            raise TypeError(
                "other must be type Vector with the same length, "
                f"float, or int, but got {type(other)}."
            )
        return Vector(self._res_buff)

    def __truediv__(self, other: float | int) -> Vector:
        vector_scalar_div(self._vals, other, self._res_buff)
        return Vector(self._res_buff)

    def __str__(self) -> str:
        return str(vectord_to_list(self._vals))
