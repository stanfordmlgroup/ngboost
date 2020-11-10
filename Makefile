
install:
	pip install poetry && poetry install

package:
	poetry build

publish: package
	poetry publish

lint:
	poetry run pre-commit run --hook-stage manual --all-files

test:
	poetry run pytest --slow -v

clean:
	rm -r build dist ngboost.egg-info
