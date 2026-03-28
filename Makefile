.PHONY: fmt lint lint-fix sanity sanity-with-fix pre-commit-install pre-commit-run

fmt:
	uv run --with ruff ruff format .

lint:
	uv run --with ruff ruff check .

lint-fix:
	uv run --with ruff ruff check . --fix

sanity: fmt lint

sanity-with-fix: fmt lint-fix

pre-commit-install:
	uv run --with pre-commit pre-commit install -t pre-commit -t pre-push

pre-commit-run:
	uv run --with pre-commit pre-commit run -a || true
