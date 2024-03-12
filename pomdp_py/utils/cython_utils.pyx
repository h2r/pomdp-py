"""Utility functions for Cython code."""

import hashlib

cpdef det_dict_hash(dct, keep=9):
    """Deterministic hash of a dictionary without sorting."""
    hash_accumulator = 0
    for key, value in dct.items():
        pair_str = f"{key}:{value}".encode()
        pair_hash = hashlib.sha1(pair_str).hexdigest()
        hash_accumulator += int(pair_hash, 16)

    # Convert the accumulated hash back to a string, take the first 'keep' digits, and convert to an integer
    hashcode = int(str(hash_accumulator)[:keep])
    return hashcode
