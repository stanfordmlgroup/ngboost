.PHONY: help install html serve clean

help:
	@echo "Please use 'make <target>' where <target> is one of:"
	@echo "  install     to install ngboost, Jupyter Book and the notebook packages"
	@echo "              (Jupyter Book also needs Node.js 18, 20 or 22+)"
	@echo "  html        to run the notebooks and build the static site into _build/html/"
	@echo "              (set BASE_URL=/ngboost to match GitHub Pages)"
	@echo "  serve       to preview the site locally with live reload"
	@echo "  clean       to remove the build files"

install:
	pip install .. -r requirements.txt

html:
	jupyter-book build --html --strict --execute

serve:
	jupyter-book start

clean:
	jupyter-book clean --all --yes
