#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 10:57:36 2017

Source:
https://abhijit-2592.github.io/Keras-with-TFODAPI/

@author: abhijit
"""

import tensorflow as tf
import tensorflow.contrib.slim as slim # only for dropout, because you need different behaviour while training and testing
from object_detection.meta_architectures import faster_rcnn_meta_arch

# Define names similar to keras from tf.keras so that ou can copy paste your model code :P
Conv2D = tf.keras.layers.Conv2D
MaxPooling2D = tf.keras.layers.MaxPooling2D
Dense = tf.keras.layers.Dense

class FasterRCNNVGG16FeatureExtractor(faster_rcnn_meta_arch.FasterRCNNFeatureExtractor):
    """VGG-16 Faster RCNN Feature Extractor
    """
    def __init__(self,
               is_training,
               first_stage_features_stride,
               batch_norm_trainable=False,
               reuse_weights=None,
               weight_decay=0.0):

        super(FasterRCNNVGG16FeatureExtractor, self).__init__(
        is_training, first_stage_features_stride, batch_norm_trainable,
        reuse_weights, weight_decay)

    def preprocess(self, resized_inputs):

        """Faster R-CNN VGG-16 preprocessing.

        mean subtraction as described here:
        https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md

        Args:
          resized_inputs: A [batch, height_in, width_in, channels] float32 tensor
            representing a batch of images with values between 0 and 255.0.

        Returns:
          preprocessed_inputs: A [batch, height_out, width_out, channels] float32
            tensor representing a batch of images.

        """
        #  imagenet bgr mean values 103.939, 116.779, 123.68, taken from keras.applications
        channel_means = [123.68, 116.779, 103.939]
        return resized_inputs - [[channel_means]]

    def _extract_proposal_features(self, preprocessed_inputs, scope):
        """Extracts first stage RPN features.

        Args:
          preprocessed_inputs: A [batch, height, width, channels] float32 tensor
            representing a batch of images.
          scope: A scope name. (unused)

        Returns:
          rpn_feature_map: A tensor with shape [batch, height, width, depth]

        NOTE:
            Make sure the naming are similar wrt to keras else creates problem while loading weights
        """
        # Block 1
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(preprocessed_inputs)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

        # Block 2
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

        # Block 3
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

        # Block 4
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

        # Block 5
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)

        return(x)


    def _extract_box_classifier_features(self, proposal_feature_maps, scope):
        """Extracts second stage box classifier features

        Args:
        proposal_feature_maps: A 4-D float tensor with shape
        [batch_size * self.max_num_proposals, crop_height, crop_width, depth]
        representing the feature map cropped to each proposal.
        scope: A scope name (unused).

        Returns:
        proposal_classifier_features: A 4-D float tensor with shape
        [batch_size * self.max_num_proposals, height, width, depth]
        representing box classifier features for each proposal.

        Use tf.slim for dropout because you need different behaviour while training and testing
        """
        x = Dense(4096, activation='relu', name='fc1')(proposal_feature_maps)
        x = slim.dropout(x, 0.5, scope="Dropout_1", is_training=self._is_training)
        x = Dense(4096, activation='relu', name='fc2')(x)
        proposal_classifier_features = slim.dropout(x, 0.5, scope="Dropout_2", is_training=self._is_training)

        return(proposal_classifier_features)

