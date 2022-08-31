import argparse
import cv2
from PIL import Image
import numpy as np
import os
import traceback


def swap_colors(source_path, target_path):
    """
    Swaps background and any other color. Only to be used on binary images.

    :param source_path: the image to load
    :type source_path: str
    :param target_path: the image to write to
    :type target_path: str
    """
    image_file = cv2.imread(source_path, cv2.IMREAD_UNCHANGED)
    seg = np.array(image_file)
    seg = np.where(seg > 0, 2, seg)
    seg = np.where(seg < 1, 1, seg)
    seg = np.where(seg > 1, 0, seg)
    cv2.imwrite(target_path, seg)


def convert_image(source_path, target_path):
    """
    Converts a single image.

    :param source_path: the input image to read
    :type source_path: str
    :param target_path: the output image to write
    :type target_path: str
    """
    image_file = Image.open(source_path)
    seg = np.array(image_file)
    seg = seg.astype(np.uint8)
    seg2 = np.zeros((seg.shape[0], seg.shape[1], 3))
    seg2[:, :, 0] += seg[:, :]
    cv2.imwrite(target_path, seg2)


def convert(in_dir, out_dir, dry_run=False, swap=False, verbose=False):
    """
    Converts PNGs in the input directory and stores them in the output directory.

    :param in_dir: the input directory
    :type in_dir: str
    :param out_dir: the output directory
    :type out_dir: str
    :param dry_run: whether to perform only a dry-run, not actually generate any output
    :type dry_run: bool
    :param swap: whether to swap background/foreground colors (binary images only!)
    :type swap: bool
    :param verbose: whether to output some progress information
    :type verbose: bool
    """

    for f in os.listdir(in_dir):
        if f.lower().endswith(".png"):
            if verbose:
                print(f)
            in_file = os.path.join(in_dir, f)
            if not dry_run:
                out_file = os.path.join(out_dir, f)
                if swap:
                    swap_colors(in_file, out_file)
                    convert_image(out_file, out_file)
                else:
                    convert_image(in_file, out_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Converts indexed PNGs into PNGs with the indices in the blue channel.")
    parser.add_argument('-i', '--input_dir', help='Directory with indexed PNGs to convert', required=True, default=None)
    parser.add_argument('-o', '--output_dir', help='Directory to stored the converted PNGs in', required=False, default=None)
    parser.add_argument('-s', '--swap', action='store_true', help='Whether to swap background and foreground colors (only use on binary images!)', required=False, default=False)
    parser.add_argument('-n', '--dry_run', action='store_true', help='Whether to perform only a dry-run, not actually generate output files', required=False, default=False)
    parser.add_argument('-v', '--verbose', action='store_true', help='Whether to output some progress information', required=False, default=False)
    parsed = parser.parse_args()
    if (not parsed.dry_run) and (parsed.output_dir is None):
        raise Exception("No output directory provided!")

    try:
        convert(parsed.input_dir, parsed.output_dir,
                dry_run=parsed.dry_run, swap=parsed.swap, verbose=parsed.verbose)
    except:
        print(traceback.format_exc())
