# setup.py
# Copyright (C) 2019 Fracpete (fracpete at waikato dot ac dot nz)

from setuptools import setup, find_namespace_packages


setup(
    name="wai.tfimageclass",
    description="Image classification using tensorflow.",
    url="https://github.com/waikato-datamining/tensorflow/image_classification",
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
    packages=find_namespace_packages(where='src'),
    namespace_packages=[
        "wai",
    ],
    version="0.0.1",
    author='Peter Reutemann and TensorFlow Team',
    author_email='fracpete@waikato.ac.nz',
    install_requires=[
        "argparse",
        "numpy",
        "pillow",
        "tensorflow_hub",
    ],
    entry_points={
        "console_scripts": [
            "tfic-retrain=wai.tfimageclass.train.retrain:sys_main",
            "tfic-labelimage=wai.tfimageclass.predict.label_image:sys_main",
            "tfic-poll=wai.tfimageclass.predict.poll:sys_main",
        ]
    }
)
