# pomdp-py

Python POMDP library with Cython algorithm implementation.

Refer to the [docs](https://h2r.github.io/pomdp-py/html/).

## Development

For development, please run
```
pip install -e .
```
to build and install this package. This will build `.so` files and copy them to the python source directory.
When you make changes to `.pyx` or `.pyd` files, run
```
make build
```
to rebuild those `.so` libraries, so that the python imports can get those changes you made.
