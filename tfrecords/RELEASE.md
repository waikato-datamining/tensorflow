## Update

* Add any changes for this version to ``CHANGES.rst``.

* Update ``setup.py`` to include new version number. Should match above.

* Commit/push.

## Release on PyPI

* Delete the previous artefacts:

```
rm -r dist src/wai.tfrecords.egg-info
```

* Create new release

```
python setup.py clean
python setup.py sdist
```

* Release

```
twine upload dist/*
```

## Publish the release on GitHub

* Start a new release

* Make the tag ``wai.tfrecords-vX.Y.Z``

* Title: ``wai.tfrecords vX.Y.Z``

* Copy change-log entry for release into description

* Upload all binaries in ``dist/``

* Publish 
