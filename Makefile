.PHONY: test clean build deploy test_deploy

test:
	python setup.py test

clean:
	rm -rf build tmp dist *.egg-info *.so

build:
	python setup.py sdist bdist_wheel

deploy: clean build
	twine upload dist/*

test_deploy: clean build
	twine upload --repository-url https://test.pypi.org/legacy/ dist/*	
