import argparse
import traceback
from keras_segmentation.train import train
from keras_segmentation.models.all_models import model_from_name
from keras_segmentation.data_utils.augmentation import augmentation_functions

try:
    import imgaug as ia
    from imgaug import augmenters as iaa
except ImportError:
    print("Error in loading augmentation, can't import imgaug."
          "Please make sure it is installed.")


def _load_augmentation_aug_simple():
    """ Load image augmentation model 'simple' """

    def sometimes(aug):
        return iaa.Sometimes(0.5, aug)

    return iaa.Sequential(
        [
            # apply the following augmenters to most images
            iaa.Fliplr(0.5),  # horizontally flip 50% of all images
            iaa.Flipud(0.2),  # vertically flip 20% of all images
            # crop images by -5% to 10% of their height/width
            sometimes(iaa.CropAndPad(
                percent=(-0.05, 0.1),
                pad_mode='constant',
                pad_cval=(0, 255)
            )),
            sometimes(iaa.Affine(
                # scale images to 80-120% of their size, individually per axis
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                # translate by -20 to +20 percent (per axis)
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                rotate=(-45, 45),  # rotate by -45 to +45 degrees
                shear=(-16, 16),  # shear by -16 to +16 degrees
                # use nearest neighbour or bilinear interpolation (fast)
                order=[0, 1],
                # if mode is constant, use a cval between 0 and 255
                cval=(0, 255),
                # use any of scikit-image's warping modes
                # (see 2nd image from the top for examples)
                mode='constant'
            )),
            # execute 0 to 5 of the following (less important) augmenters per
            # image don't execute all of them, as that would often be way too
            # strong
            iaa.SomeOf((0, 5),
                       [
                # convert images into their superpixel representation
                sometimes(iaa.Superpixels(
                    p_replace=(0, 1.0), n_segments=(20, 200))),
                iaa.OneOf([
                    # blur images with a sigma between 0 and 3.0
                    iaa.GaussianBlur((0, 3.0)),
                    # blur image using local means with kernel sizes
                    # between 2 and 7
                    iaa.AverageBlur(k=(2, 7)),
                    # blur image using local medians with kernel sizes
                    # between 2 and 7
                    iaa.MedianBlur(k=(3, 11)),
                ]),
                iaa.Sharpen(alpha=(0, 1.0), lightness=(
                            0.75, 1.5)),  # sharpen images
                #iaa.Emboss(alpha=(0, 1.0), strength=(
                #    0, 2.0)),  # emboss images
                # search either for all edges or for directed edges,
                # blend the result with the original image using a blobby mask
                iaa.SimplexNoiseAlpha(iaa.OneOf([
                    iaa.EdgeDetect(alpha=(0.5, 1.0)),
                    iaa.DirectedEdgeDetect(
                        alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                ])),
                # add gaussian noise to images
                iaa.AdditiveGaussianNoise(loc=0, scale=(
                    0.0, 0.05*255), per_channel=0.5),
                iaa.OneOf([
                    # randomly remove up to 10% of the pixels
                    iaa.Dropout((0.01, 0.1), per_channel=0.5),
                    iaa.CoarseDropout((0.03, 0.15), size_percent=(
                        0.02, 0.05), per_channel=0.2),
                ]),
                # invert color channels
                #iaa.Invert(0.05, per_channel=True),
                # change brightness of images (by -10 to 10 of original value)
                iaa.Add((-10, 10), per_channel=0.5),
                # change hue and saturation
                iaa.AddToHueAndSaturation((-20, 20)),
                # either change the brightness of the whole image (sometimes
                # per channel) or change the brightness of subareas
                iaa.OneOf([
                    iaa.Multiply(
                                (0.5, 1.5), per_channel=0.5),
                    iaa.FrequencyNoiseAlpha(
                        exponent=(-4, 0),
                        first=iaa.Multiply(
                            (0.5, 1.5), per_channel=True),
                        second=iaa.ContrastNormalization(
                            (0.5, 2.0))
                    )
                ]),
                # improve or worsen the contrast
                iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),
                #iaa.Grayscale(alpha=(0.0, 1.0)),
                # move pixels locally around (with random strengths)
                #sometimes(iaa.ElasticTransformation(
                #    alpha=(0.5, 3.5), sigma=0.25)),
                # sometimes move parts of the image around
                #sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))),
                #sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
            ],
                random_order=True
            )
        ],
        random_order=True
    )


augmentation_functions['aug_simple'] = _load_augmentation_aug_simple


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Like the default 'train' sub-command, but with more options exposed.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model_name", type=str, required=True, help="The name of the pretrained model to use: %s" % str(', '.join(sorted(list(model_from_name.keys())))))
    parser.add_argument("--train_images", type=str, required=True, help="The directory with the training images.")
    parser.add_argument("--train_annotations", type=str, required=True, help="The directory with the training annotations.")
    parser.add_argument("--n_classes", type=int, required=True, help="The number of classes, including background.")
    parser.add_argument("--input_height", type=int, default=None, help="The height to scale the images to (multiple of 32).")
    parser.add_argument("--input_width", type=int, default=None, help="The width to scale the images to (multiple of 32)")
    parser.add_argument('--not_verify_dataset', action='store_false', help="Whether to skip verifying the dataset.")
    parser.add_argument("--checkpoints_path", type=str, default=None, help="The directory to store the checkpoint files and _config.json in.")
    parser.add_argument("--epochs", type=int, default=5, help="The number of epochs to train.")
    parser.add_argument("--batch_size", type=int, default=2, help="The batch size.")
    parser.add_argument('--validate', action='store_true', help="Whether to validate using a separate validation set.")
    parser.add_argument("--val_images", type=str, default="", help="The images of the validation set.")
    parser.add_argument("--val_annotations", type=str, default="", help="The annotations of the validation set.")
    parser.add_argument("--val_batch_size", type=int, default=2, help="The batch size for the validation.")
    parser.add_argument("--load_weights", type=str, default=None, help="The pretrained model to use.")
    parser.add_argument('--auto_resume_checkpoint', action='store_true', help="Whether to resume from the last checkpoint available from the checkpoints directory.")
    parser.add_argument("--steps_per_epoch", type=int, default=512, help="The number of steps per epoch.")
    parser.add_argument("--val_steps_per_epoch", type=int, default=512, help="The number of steps per validation epoch.")
    parser.add_argument('--use_multi_processing', action='store_true', help="Whether to use process-based multi-processing.")
    parser.add_argument('--ignore_zero_class', action='store_true', help="Defines the loss function, either 'categorical_crossentropy' or 'masked_categorical_crossentropy' (if ignoring zero class)")
    parser.add_argument("--optimizer_name", type=str, default="adadelta", help="The name of the optimizer to use (https://www.tensorflow.org/versions/r1.14/api_docs/python/tf/keras/optimizers).")
    parser.add_argument('--do_augment', action='store_true', help="Whether to augment the images.")
    parser.add_argument("--augmentation_name", type=str, default="aug_simple", help="How to augment the images: aug_all, aug_all2, aug_simple, aug_geometric, aug_non_geometric")
    args = parser.parse_args()

    try:
        train(model=args.model_name,
              train_images=args.train_images,
              train_annotations=args.train_annotations,
              input_height=args.input_height,
              input_width=args.input_width,
              n_classes=args.n_classes,
              verify_dataset=args.not_verify_dataset,
              checkpoints_path=args.checkpoints_path,
              epochs=args.epochs,
              batch_size=args.batch_size,
              validate=args.validate,
              val_images=args.val_images,
              val_annotations=args.val_annotations,
              val_batch_size=args.val_batch_size,
              auto_resume_checkpoint=args.auto_resume_checkpoint,
              load_weights=args.load_weights,
              steps_per_epoch=args.steps_per_epoch,
              val_steps_per_epoch=args.val_steps_per_epoch,
              optimizer_name=args.optimizer_name,
              gen_use_multiprocessing=args.use_multi_processing,
              ignore_zero_class=args.ignore_zero_class,
              do_augment=args.do_augment,
              augmentation_name=args.augmentation_name)
    except Exception as e:
        print(traceback.format_exc())
