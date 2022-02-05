# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="statistics_lib", # Replace with your own username
    version="0.1.0",
    author="Martin GARAJ",
    author_email="garaj.martin@gmail.com",
    description="A python library to ease some use of statistical tools.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/martin-garaj/statistics",
    packages=setuptools.find_packages(),
    package_dir={'statistics_lib': 'statistics_lib'},
    package_data={},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
