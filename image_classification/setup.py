# setup.py
# Copyright (C) 2019 Fracpete (fracpete at waikato dot ac dot nz)

from setuptools import setup


setup(
    name="tfrecords",
    description="Image classification using tensorflow.",
    url="https://github.com/waikato-datamining/tensorflow/image_classification",
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Programming Language :: Python :: 3',
    ],
    license='MIT License',
    packages=[
        "wai.tf.imageclassification",
    ],
    version="0.0.1",
    author='Peter Reutemann',
    author_email='fracpete@waikato.ac.nz',
    install_requires=[
        "argparse",
        "numpy",
        "pillow",
    ],
)
