.PHONY: test

clean:
	find . -name '__pycache__' -exec rm -rf {} +
	find . -name '.pytest_cache' -exec rm -rf {} +

test:
	export PYTHONPATH=$(shell pwd) && pytest
