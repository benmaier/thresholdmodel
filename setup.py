from setuptools import setup, Extension
import setuptools
import os, sys

# get __version__, __author__, and __email__
exec(open("./thresholdmodel/metadata.py").read())

setup(
    name = 'thresholdmodel',
    version = __version__,
    author = __author__,
    author_email = __email__,
    url = 'https://github.com/benmaier/thresholdmodel',
    license = __license__,
    description = "Simulate a continuous-time threshold model on static networks.",
    long_description = '',
    packages = setuptools.find_packages(),
    setup_requires = [
            ],
    install_requires = [
                'networkx>=2.0',
                'numpy>=1.14',
            ],
    include_package_data = True,
    zip_safe = False,
)
