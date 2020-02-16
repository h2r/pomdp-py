clean:
	rm -rf build/

.PHONY: build
build:
	python setup.py build_ext --inplace



