# PyPi

Preparation:

* increment version in `setup.py`
* add new changelog section in `CHANGES.rst`
* commit/push all changes

Commands for releasing on pypi.org (requires twine >= 1.8.0):

```
find -name "*~" -delete
rm dist/*
./venv/bin/python setup.py clean
./venv/bin/python setup.py sdist
./venv/bin/twine upload dist/*
```


# Github

Steps:

* start new release (version: `wai.tflite_model_maker-vX.Y.Z`)
* enter release notes, i.e., significant changes since last release
* upload `wai.tflite_model_maker-X.Y.Z.tar.gz` previously generated with `setup.py`
* publish


# Docker
If necessary, update the docker images for CPU and GPU and deploy them
(remove them from the local registry before deploying them).
