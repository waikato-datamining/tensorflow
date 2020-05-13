# setup.py
# Copyright (C) 2020 Fracpete (fracpete at waikato dot ac dot nz)

from setuptools import setup, find_namespace_packages

def _read(f) -> bytes:
    """
    Reads in the content of the file.
    :param f: the file to read
    :type f: str
    :return: the content
    :rtype: str
    """
    return open(f, 'rb').read()


setup(
    name="wai.tfutils",
    description="Library with utility functions to reduce amount of code for making predictions with TensorFlow.",
    long_description=(
        _read('DESCRIPTION.rst') + b'\n' +
        _read('CHANGES.rst')).decode('utf-8'),
    url="https://github.com/waikato-datamining/tensorflow/tree/master/tfutils",
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: Apache Software License',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Programming Language :: Python :: 3',
    ],
    license='Apache 2.0 License',
    package_dir={
        '': 'src'
    },
    packages=find_namespace_packages(where='src'),
    namespace_packages=[
        "wai",
    ],
    version="0.0.3",
    author='Peter Reutemann and TensorFlow Team',
    author_email='fracpete@waikato.ac.nz',
    install_requires=[
        "numpy",
        "pillow",
    ],
)
