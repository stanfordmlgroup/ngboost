import os
import setuptools


def get_version() -> str:
    version_filepath = os.path.join(os.path.dirname(__file__), "ngboost", "version.py")
    with open(version_filepath) as f:
        for line in f:
            if line.startswith("__version__"):
                return line.strip().split()[-1][1:-1]
    assert False


def get_long_description():
    readme_path = os.path.join(os.path.dirname(__file__), "README.md")
    with open(readme_path) as f:
        return f.read()


setuptools.setup(
    name="ngboost",
    version=get_version(),
    author="Stanford ML Group",
    author_email="avati@cs.stanford.edu",
    description="Library for probabilistic predictions via gradient boosting.",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/stanfordmlgroup/ngboost",
    license="Apache License 2.0",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "numpy>=1.17.2",
        "scipy>=1.3.1",
        "scikit-learn>=0.21.3",
        "tqdm>=4.36.1",
        "lifelines>=0.22.8",
    ],
    tests_require=["pytest", "pre-commit", "black"],
)
