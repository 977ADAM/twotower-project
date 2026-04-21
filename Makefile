.PHONE:

run-lint:
	@ruff check src tests && \
	mypy src tests \
