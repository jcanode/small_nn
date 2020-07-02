import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="example-pkg-jcanode",
    version="0.0.1",
    author="Justin Canode",
    author_email="jcanode@my.gcu.edu",
    description="A small Neural Network Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jcanode/small_nn",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
