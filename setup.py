import setuptools

setuptools.setup(
    name="ngboost",
    version="0.2.0",
    author="Stanford ML Group",
    author_email="avati@cs.stanford.edu",
    description="Library for probabilistic predictions via gradient boosting.",
    long_description="Please see Github for full description.",
    long_description_content_type="text/markdown",
    url="https://github.com/stanfordmlgroup/ngboost",
    license='Apache License 2.0',
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
    tests_require=[
        "pytest"
    ]
)
