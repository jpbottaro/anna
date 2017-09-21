venv: venv/bin/activate

venv/bin/activate:
	@test -d venv || python3 -m venv --clear venv
	venv/bin/python -m pip install -Ur requirements.txt
	@touch venv/bin/activate

datasets: venv
	venv/bin/python anna/datasets/main.py

test: venv
	venv/bin/python setup.py test

clean:
	rm -rf venv

.PHONY: venv datasets test clean
