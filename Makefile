all:
	python3 -m pylint ngboost
pkg:
	python3 setup.py sdist bdist_wheel
clean:
	rm -r build dist ngboost.egg-info
upload:
	twine upload dist/*
