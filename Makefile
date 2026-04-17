.PHONY: install install-dev test test-fast demo synthetic-eval compile space-bundle

install:
	python -m pip install -e .

install-dev:
	python -m pip install -r requirements-dev.txt
	python -m pip install -r space/requirements.txt
	python -m pip install -e .

test:
	python -m unittest discover -s tests -p 'test_*.py' -v

test-fast:
	python -m unittest discover -s tests -p 'test_frontier_memory.py' -v

demo:
	python space/app.py

synthetic-eval:
	python scripts/run_synthetic_eval.py --candidate candidates/bootstrap_v0.yaml --dataset-size 8 --seed 7

compile:
	python -m compileall frontier_memory scripts space tests

space-bundle:
	bash scripts/deploy_space.sh
