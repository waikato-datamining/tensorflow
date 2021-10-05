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
    name="wai.tflite_model_maker",
    description="tflite model maker command-line utilities.",
    long_description=(
        _read('DESCRIPTION.rst') + b'\n' +
        _read('CHANGES.rst')).decode('utf-8'),
    url="https://github.com/waikato-datamining/tensorflow/tree/master/tflite_model_maker",
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
    version="0.0.1",
    author='Peter Reutemann and TensorFlow Team',
    author_email='fracpete@waikato.ac.nz',
    install_requires=[
        "argparse",
        "numpy<1.20.0",
        "tflite-model-maker>=0.3.0,<0.3.2",
        "tensorflow>=2.4.0,<2.5.0",
        "wai.pycocotools",
        "pyyaml",
        "opex",
        "python-image-complete",
        "simple-file-poller>=0.0.9",
        "redis",
        "redis-docker-harness==0.0.1",
    ],
    entry_points={
        "console_scripts": [
            "tmm-od-train=wai.tmm.objdet.train:sys_main",
            "tmm-od-predict=wai.tmm.objdet.predict:sys_main",
            "tmm-od-predict-poll=wai.tmm.objdet.predict_poll:sys_main",
            "tmm-od-predict-redis=wai.tmm.objdet.predict_redis:sys_main",
        ]
    }
)
