
install:
	pip install poetry==1.2.1 && poetry install

package:
	poetry build

publish: package
	source .env && poetry config pypi-token.pypi $$PYPI_TOKEN && poetry publish

lint:
	poetry run pre-commit run --hook-stage manual --all-files

test:
	poetry run pytest --slow -v

clean:
	rm -rf dist/*
