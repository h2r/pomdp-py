"""Utility functions for Cython code."""

import hashlib

cpdef det_dict_hash(dct, keep=9):
    """deterministic hash of a dictionary."""
    content = str(list(sorted(dct.items()))).encode()
    hashcode = int(str(int(hashlib.sha1(content).hexdigest(), 16))[:keep])
    return hashcode
