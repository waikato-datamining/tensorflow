import argparse
import cv2
from PIL import Image
import numpy
import os
import traceback


def convert_image(source_path, target_path):
    """
    Converts a single image.

    :param source_path: the input image to read
    :type source_path: str
    :param target_path: the output image to write
    :type target_path: str
    """
    image_file = Image.open(source_path)
    seg = numpy.array(image_file)
    seg = seg.astype(numpy.uint8)
    seg2 = numpy.zeros((seg.shape[0], seg.shape[1], 3))
    seg2[:, :, 0] += seg[:, :]
    cv2.imwrite(target_path, seg2)


def convert(in_dir, out_dir, verbose=False):
    """
    Converts PNGs in the input directory and stores them in the output directory.

    :param in_dir: the input directory
    :type in_dir: str
    :param out_dir: the output directory
    :type out_dir: str
    :param verbose: whether to output some progress information
    :type verbose: bool
    """

    for f in os.listdir(in_dir):
        if f.lower().endswith(".png"):
            if verbose:
                print(f)
            in_file = os.path.join(in_dir, f)
            out_file = os.path.join(out_dir, f)
            convert_image(in_file, out_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Converts indexed PNGs into PNGs with the indices in the blue channel.")
    parser.add_argument('--input_dir', help='Directory with indexed PNGs to convert', required=True, default=None)
    parser.add_argument('--output_dir', help='Directory to stored the converted PNGs in', required=True, default=None)
    parser.add_argument('--verbose', action='store_true', help='Whether to output some progress information', required=False, default=False)
    parsed = parser.parse_args()

    try:
        convert(parsed.input_dir, parsed.output_dir, verbose=parsed.verbose)
    except:
        print(traceback.format_exc())
