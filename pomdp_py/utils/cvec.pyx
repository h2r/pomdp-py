# cython: language_level=3, boundscheck=False, wraparound=False, profile=True

from __future__ import annotations
from libc.math cimport fmin, fmax
import numpy as np
cimport numpy as cnp
from typing import Iterator, Iterable
cnp.import_array()

ArrayDtype_t = np.float64

cdef inline Arrayf_t null_vector(unsigned int n_zeros):
    return np.zeros((n_zeros,), ArrayDtype_t)


cpdef inline Arrayf_t list_to_vectord(list[float] values):
    return np.array(values, ArrayDtype_t)


cpdef inline list[float] vectord_to_list(Arrayf_t values):
    return values.tolist()


cdef inline bint vectors_are_not_same_size(double[:] v0, double[:] v1):
    return v0.shape[0] != v1.shape[0]


cdef inline bint vector_size_is_zero(double[:] v):
    return v.shape[0] == 0


cdef double vector_dot_prod(double[:] v0, double[:] v1):
    cdef unsigned int i = 0
    cdef double _sum = 0.

    if vectors_are_not_same_size(v0, v1):
        raise ValueError("Both vectors must have the same size.")
    if vector_size_is_zero(v0):
        raise ValueError("Vectors should contain at least one value.")

    for i in range(v0.shape[0]):
        _sum += (v0[i] * v1[i])
    return _sum


cdef void vector_add(double[:] v0, double[:] v1, double[:] res):
    if vectors_are_not_same_size(v0, v1):
        raise ValueError("Both vectors must have the same size.")
    if vector_size_is_zero(v0):
        raise ValueError("Vectors should contain at least one value.")

    cdef unsigned int i = 0
    for i in range(v0.shape[0]):
        res[i] = v0[i] + v1[i]


cdef void vector_adds(double[:] v, double scalar, double[:] res):
    if vector_size_is_zero(v):
        raise ValueError("Vector should contain at least one value.")

    cdef unsigned int i = 0
    for i in range(v.shape[0]):
        res[i] = v[i] + scalar


cdef void vector_muls(double[:] v, double scalar,double[:] res):
    if vectors_are_not_same_size(v, res):
        raise ValueError("Vectors v and res must be the same size.")
    if vector_size_is_zero(v):
        raise ValueError("Vector should contain at least one value.")

    cdef unsigned int i = 0
    for i in range(v.shape[0]):
        res[i] = v[i] * scalar


cdef void vector_sub(double[:] v0, double[:] v1, double[:] res):
    if vectors_are_not_same_size(v0, v1):
        raise ValueError("Both vectors must have the same size.")
    if vector_size_is_zero(v0):
        raise ValueError("Vectors should contain at least one value.")

    cdef unsigned int i = 0
    for i in range(v0.shape[0]):
        res[i] = v0[i] - v1[i]


cdef void vector_subvs(double[:] v, double scalar, double[:] res):
    if vector_size_is_zero(v):
        raise ValueError("Vector should contain at least one value.")

    cdef unsigned int i = 0
    for i in range(v.shape[0]):
        res[i] = v[i] - scalar


cdef void vector_subsv(double scalar, double[:] v, double[:] res):
    if vector_size_is_zero(v):
        raise ValueError("Vector should contain at least one value.")

    cdef unsigned int i = 0
    for i in range(v.shape[0]):
        res[i] = scalar - v[i]


cdef void vector_scalar_div(double[:] v, double scalar, double[:] res):
    if vector_size_is_zero(v):
        raise ValueError("Vector should contain at least one value.")
    if scalar == 0.0:
        raise ZeroDivisionError("Scalar division by zero!")

    cdef unsigned int i = 0
    for i in range(v.shape[0]):
        res[i] = v[i] / scalar

cdef unsigned int vector_argmax(double[:] v):
    cdef int n_values = v.shape[0]
    if vector_size_is_zero(v):
        raise ValueError("Vector should contain at least one value.")
    if n_values == 1:
        return 0

    cdef int max_idx = 0
    cdef int i = 0
    for i in range(1, n_values):
        if v[i] > v[max_idx]:
            max_idx = i
    return max_idx


cdef unsigned int vector_argmin(double[:] v):
    cdef int n_values = v.shape[0]
    if vector_size_is_zero(v):
        raise ValueError("Vector should contain at least one value.")
    if n_values == 1:
        return 0

    cdef int min_idx = 0
    cdef int i = 0
    for i in range(1, n_values):
        if v[i] < v[min_idx]:
            min_idx = i
    return min_idx


cdef void vector_clip(double[:] v, double min_value, double max_value):
    cdef int n_values = v.shape[0]
    if vector_size_is_zero(v):
        raise ValueError("Vector should contain at least one value.")
    if min_value >= max_value:
        raise ValueError(
            f"Min value ({min_value}) must be less than max value ({max_value})."
        )
    cdef int i = 0
    for i in range(n_values):
        v[i] = fmax(min_value, fmin(max_value, v[i]))


cdef void vector_copy(double[:] src, double[:] dst):
    if vector_size_is_zero(src):
        raise ValueError("Vector should contain at least one value.")
    cdef int i = 0
    for i in range(src.shape[0]):
        dst[i] = src[i]


cdef void vector_resize(Arrayf_t v, unsigned int new_size):
    if new_size <= 0:
        raise ValueError("New vector size must be a positive integer.")
    v = np.zeros((new_size,), dtype=v.dtype)


cdef class Vector:
    """
    The Vector class. Provides an implementation of a vector for
    maintaining multiple values.
    """

    def __init__(self, values: Iterable[float] = (0.0,)):
        # Perform a lazy conversion of the input values.
        self._vals = list_to_vectord(list(values)).flatten()
        self._res_buff = null_vector(self._vals.shape[0])
        self._length = self._vals.shape[0]

    cdef bint _index_is_out_of_range(Vector self, unsigned int index):
        return index < 0 or self._length <= index

    def as_list(self) -> list[float]:
        """
        Returns a list of the internal values.
        """
        return vectord_to_list(self._vals)

    def as_vector(self) -> np.ndarray:
        return self._vals[:]

    cpdef void clip(Vector self, double min_value, double max_value):
        """
        Clips the values within the value using the given min and max values.
        """
        vector_clip(self._vals, min_value, max_value)

    cpdef Vector copy(Vector self):
        """
        Returns a copy of this vector.
        """
        return Vector(self._vals)

    cpdef double dot(Vector self, Vector other):
        """
        Performs the dot product between two Vectors.
        """
        return vector_dot_prod(self._vals, other._vals)

    @staticmethod
    def fill(value: float, n_values: int) -> Vector:
        return Vector([value] * n_values)

    cpdef int len(Vector self):
        return self._length

    cpdef unsigned int argmax(Vector self):
        return vector_argmax(self._vals)

    cpdef unsigned int argmin(Vector self):
        return vector_argmin(self._vals)

    cpdef double max(Vector self):
        return self._vals[self.argmax()]

    cpdef double min(Vector self):
        return self._vals[self.argmin()]

    @staticmethod
    def null(n_zeros: int) -> Vector:
        return Vector.fill(0.0, n_zeros)

    cdef void resize(Vector self, unsigned int new_size):
        self._vals = null_vector(new_size)
        self._res_buff = null_vector(new_size)
        self._length = self._vals.shape[0]

    cpdef void zeros(Vector self):
        cdef int i
        if self._length == 1:
            self._vals[0] = 0.
        else:
            for i in range(self._length):
                self._vals[i] = 0.

    cpdef double get(Vector self, unsigned int index):
        if self._index_is_out_of_range(index):
            raise IndexError(
                f"index ({index}) is out-of-range for length {self._length}."
            )
        return self._vals[index]

    cpdef void set(Vector self, unsigned int index, double value):
        if self._index_is_out_of_range(index):
            raise IndexError(
                f"index ({index}) is out-of-range for length {self._length}."
            )
        self._vals[index] = <double> value

    def __getitem__(self, index: int) -> float:
        return self.get(index)

    def __setitem__(self, index: int, value: float) -> None:
        self.set(index, value)

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
