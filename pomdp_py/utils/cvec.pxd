# cython: language_level=3

from __future__ import annotations
from libcpp.vector cimport vector

ctypedef vector[double] vectord_t


cdef vectord_t null_vector(unsigned int n_zeros) except *
cpdef vectord_t list_to_vectord(list[float] values)
cpdef list[float] vectord_to_list(vectord_t values)

cdef double vector_dot_prod(const vectord_t& v0, const vectord_t& v1) except *
cdef void vector_add(const vectord_t& v0, const vectord_t& v1, vectord_t& res) except *
cdef void vector_adds(const vectord_t& v, const double& scalar, vectord_t& res) except *
cdef void vector_muls(const vectord_t& v, const double& scalar, vectord_t& res) except *
cdef void vector_sub(const vectord_t& v0, const vectord_t& v1, vectord_t& res) except *
cdef void vector_subvs(const vectord_t& v, const double& scalar, vectord_t& res) except *
cdef void vector_subsv(const double& scalar, const vectord_t& v, vectord_t& res) except *
cdef void vector_scalar_div(const vectord_t& v, const double& scalar, vectord_t& res) except *

cdef double vector_max(const vectord_t& v) except *
cdef double vector_min(const vectord_t& v) except *
cdef void vector_clip(vectord_t& v, const double& min_value, const double& max_value) except *
cdef void vector_copy(const vectord_t& src, vectord_t& dst) except *


cdef class Vector:
    cdef vectord_t _vals
    cdef vectord_t _res_buff
    cdef int _length

    cdef bint _is_in_range(Vector self, int index)
    cpdef Vector copy(Vector self)
    cpdef double dot(Vector self, Vector other)
    cpdef int len(Vector self)
    cdef double max(Vector self)
    cdef double min(Vector self)
