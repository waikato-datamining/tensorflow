import argparse
import traceback
from keras_segmentation.train import train
from keras_segmentation.models.all_models import model_from_name

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
    parser.add_argument("--augmentation_name", type=str, default="aug_all", help="How to augment the images: aug_all, aug_all2, aug_geometric, aug_non_geometric")
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
              gen_use_multiprocessing=args.use_multiprocessing,
              ignore_zero_class=args.ignore_zero_class,
              do_augment=args.do_augment,
              augmentation_name=args.augmentation_name)
    except Exception as e:
        print(traceback.format_exc())
