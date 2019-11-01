Image classification (not object detection) using `tensorflow <https://www.tensorflow.org/>`__.

Based on example code located here:

`https://www.tensorflow.org/hub/tutorials/image_retraining <https://www.tensorflow.org/hub/tutorials/image_retraining>`__


Usage
=====

All scripts support `--help` option to list all available options.


Train
-----

- For training, use module `wai.tfimageclass.train.retrain` or console script `tfic-retrain`
- For evaluating a built model, use module `wai.tfimageclass.train.stats` or console script `tfic-stats`


Training data
-------------

All the data for building the model must be located in a single directory, with each sub-directory representing
a *label*. For instance for building a model for distinguishing flowers (daisy, dandelion, roses, sunflowers, tulip),
the data directory looks like this::

   |
   +- flowers
      |
      +- daisy
      |
      +- dandelion
      |
      +- roses
      |
      +- sunflowers
      |
      +- tulip


Predict
-------

Once you have built a model, you can use it as follows:

- For making predictions for a single image, use module `wai.tfimageclass.predict.label_image` or console
  script `tfic-labelimage`
- For polling images in a directory and making continous predictions with CSV companion files, use
  module `wai.tfimageclass.predict.poll` or console script `tfic-poll`
