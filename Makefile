.PHONY: test clean build

test:
	python -m unittest

clean:
	rm -rf build tmp dist *.egg-info *.so .eggs



build:
	pip install .

# deploy: clean build
# 	twine upload dist/*

# test_deploy: clean build
# 	twine upload --repository-url https://test.pypi.org/legacy/ dist/*	
