# setup.py
# Copyright (C) 2019 Fracpete (fracpete at waikato dot ac dot nz)

from setuptools import setup, find_namespace_packages


setup(
    name="wai.tfrecords",
    description="Converting ADAMS annotations to tfrecords.",
    url="https://github.com/waikato-datamining/tensorflow/tfrecords",
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Programming Language :: Python :: 3',
    ],
    license='MIT License',
    package_dir={
        '': 'src'
    },
    packages=find_namespace_packages(where="src"),
    namespace_packages=[
        "wai"
    ],
    version="0.0.2",
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
        "wai.common"
    ],
    entry_points={
        "console_scripts": ["tfrecords-convert=wai.tfrecords.adams:sys_main"]
    }
)
