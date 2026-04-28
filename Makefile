.PHONY: lint typecheck test

lint:
	ruff check twotower/ tests/

typecheck:
	mypy twotower/

test:
	pytest
