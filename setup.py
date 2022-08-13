# Copyright 2022 Diabwellness.ai, Inc.
# All rights reserved

"""Setup file to install the diabwellness module."""

from setuptools import setup

setup(
    name="diabwellness",
    version="0.1.0",
    description="Data analysis and Machine learning package for Diabwellness",
    url="https://github.com/cgnarendiran/diabwellness",
    author="Narendiran Chembu",
    author_email="cgnarendiran@gmail.com",
    license="MIT",
    packages=["diabwellness"],
    install_requires=["mpi4py>=2.0", "numpy"],
    classifiers=[
        "Development Status :: Under development",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3.8",
    ],
)
