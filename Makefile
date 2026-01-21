.PHONY: install test format lint clean

install:
	uv sync

test:
	uv run pytest

format:
	uv run ruff format .
	uv run ruff check --select I --fix .

lint:
	uv run ruff format . --diff
	uv run ruff check --select I .

clean:
	find . -type d -name __pycache__ -exec rm -r {} +
	find . -type f -name '*.pyc' -delete
	rm -rf jcus_link_rest.egg-info
