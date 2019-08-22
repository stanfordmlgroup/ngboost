import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ngboost",
    version="0.0.1",
    author="Tony Duan",
    author_email="tonyduan@cs.stanford.edu",
    description="See README",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tonyduan/ngboost",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
