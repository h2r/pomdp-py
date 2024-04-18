# cython: language_level=3, boundscheck=False, wraparound=False

from __future__ import annotations
cimport numpy as cnp
cnp.import_array()

ctypedef cnp.ndarray Arrayf_t


cdef Arrayf_t null_vector(unsigned int n_zeros)
cpdef Arrayf_t list_to_vectord(list[float] values)
cpdef list[float] vectord_to_list(Arrayf_t values)
cdef bint vectors_are_not_same_size(double[:] v0, double[:] v1)
cdef bint vector_size_is_zero(double[:] v)

cdef double vector_dot_prod(double[:] v0, double[:] v1) except *
cdef void vector_add(double[:] v0, double[:] v1, double[:] res) except *
cdef void vector_adds(double[:] v, double scalar, double[:] res) except *
cdef void vector_muls(double[:] v, double scalar, double[:] res) except *
cdef void vector_sub(double[:] v0, double[:] v1, double[:] res) except *
cdef void vector_subvs(double[:] v, double scalar, double[:] res) except *
cdef void vector_subsv(double scalar, double[:] v, double[:] res) except *
cdef void vector_scalar_div(double[:] v, double scalar, double[:] res) except *

cdef unsigned int vector_argmax(double[:] v) except *
cdef unsigned int vector_argmin(double[:] v) except *
cdef void vector_clip(double[:] v, double min_value, double max_value) except *
cdef void vector_copy(double[:] src, double[:] dst) except *


cdef class Vector:
    cdef cnp.ndarray _vals
    cdef cnp.ndarray _res_buff
    cdef int _length

    cdef bint _index_is_out_of_range(Vector self, unsigned int index)
    cpdef void clip(Vector self, double min_value, double max_value)
    cpdef Vector copy(Vector self)
    cpdef double dot(Vector self, Vector other)
    cpdef int len(Vector self)
    cpdef unsigned int argmax(Vector self)
    cpdef unsigned int argmin(Vector self)
    cpdef double max(Vector self)
    cpdef double min(Vector self)
    cdef void resize(Vector self, unsigned int new_size)
    cpdef void zeros(Vector self)
    cpdef double get(Vector self, unsigned int index)
    cpdef void set(Vector self, unsigned int index, double value)
