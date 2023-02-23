from pathlib import Path

from setuptools import find_packages
from setuptools import setup


requirements = ["numpy", "torch"]

current_directory = Path(__file__).parent

setup(
    author="Christopher Klugmann",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.9",
    ],
    description="Pytorch implementation of the Chernoff distance of two Dirichlet distributions.",
    install_requires=requirements,
    include_package_data=True,
    keywords=[
        "chernoff",
        "bhattacharyya",
        "distance",
        "divergence",
        "torch",
        "pytorch",
        "loss",
        "dirichlet",
        "beta"
    ],
    name="probabilistic_distance",
    packages=find_packages(),
    url="https://github.com/cklugmann/probabilistic_distance",
    version="0.0.1",
    zip_safe=False,
)