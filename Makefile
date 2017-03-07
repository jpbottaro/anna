init:
	pip install -r requirements.txt

test:
	python setup.py test

.PHONY: init test
