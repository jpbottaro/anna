export PYTHONPATH=$(PWD)

venv: .venv/bin/activate

.venv/bin/activate:
	@test -d .venv || python3 -m venv --clear .venv
	.venv/bin/python -m pip install -Ur requirements.txt
	@touch .venv/bin/activate

datasets: venv
	@.venv/bin/python anna/data/main.py data

test: venv
	@.venv/bin/python setup.py test

run: venv
	@TF_CPP_MIN_LOG_LEVEL=3 .venv/bin/python anna/main.py data reuters

run-rcv1: venv
	@TF_CPP_MIN_LOG_LEVEL=3 .venv/bin/python anna/main.py data rcv1

run-bioasq: venv
	@TF_CPP_MIN_LOG_LEVEL=3 .venv/bin/python anna/main.py data bioasq

tb: venv
	@.venv/bin/tensorboard --logdir data/model

notebook: venv
	@.venv/bin/jupyter notebook --notebook-dir=notebook

clean:
	rm -rf .venv

.PHONY: venv datasets test run run-rcv1 run-bioasq tb clean notebook
