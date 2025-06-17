dev:
	uvicorn laime:app --reload --port 2001 --host 0.0.0.0

test:
	pytest --cov=src/laime/ --cov-report lcov --cov-report term-missing --junitxml=junit.xml -o junit_family=legacy -s src/

lint:
	ruff check
	basedpyright src/laime/

alias fmt := format

format:
	ruff check --select I --fix
	ruff format
	nixfmt flake.nix
