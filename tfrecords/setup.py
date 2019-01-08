# setup.py
# Copyright (C) 2019 Fracpete (fracpete at waikato dot ac dot nz)

from setuptools import setup


setup(
    name="tfrecords",
    description="Converting ADAMS annotations to tfrecords.",
    url="https://github.com/waikato-datamining/tensorflow/tfrecords",
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Programming Language :: Python :: 3',
    ],
    license='MIT License',
    packages=[
        "adams2objectdetection",
    ],
    version="0.0.1",
    author='Peter Reutemann',
    author_email='fracpete@waikato.ac.nz',
    install_requires=[
        "cython",
        "javaproperties",
        "argparse",
        "contextlib2",
        "pillow",
        "lxml",
        "jupyter",
        "matplotlib",
        "numpy",
    ],
)
