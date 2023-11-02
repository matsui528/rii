.PHONY: test clean build

test:
	python setup.py test

clean:
	rm -rf build tmp dist *.egg-info *.so .eggs

# build:
# 	python setup.py sdist

build:
	pip install .

# deploy: clean build
# 	twine upload dist/*

# test_deploy: clean build
# 	twine upload --repository-url https://test.pypi.org/legacy/ dist/*	
