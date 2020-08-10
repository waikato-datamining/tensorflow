#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
from pydub import AudioSegment
from pydub.silence import split_on_silence
from pydub import effects


def split_audio(input_file, output_dir, min_silence_length=500, silence_threshold=-24, normalize=False, verbose=False):
    """
    Splits the MP3/OGG/WAV file into smaller chunks on pauses it identifies.

    :param input_file: the input WAV file to split
    :type input_file: str
    :param output_dir: the output directory for storing the WAV splits
    :type output_dir: str
    :param min_silence_length: the minimum length in milliseconds before considering it a pause
    :type min_silence_length: int
    :param silence_threshold: the silence threshold, i.e., if quieter than X dBFS (https://en.wikipedia.org/wiki/DBFS)
    :type silence_threshold: int
    :param normalize: whether to perform amplitude normalization
    :type normalize: bool
    :param verbose: whether to output some debugging output
    :type verbose: bool
    :return: the output files (list of strings)
    :rtype: list
    """

    if verbose:
        print("Input file: " + input_file)

    ext = os.path.splitext(input_file)[1].lower()
    if ext == ".mp3":
        sound_file = AudioSegment.from_mp3(input_file)
    elif ext == ".ogg":
        sound_file = AudioSegment.from_ogg(input_file)
    elif ext == ".wav":
        sound_file = AudioSegment.from_wav(input_file)
    else:
        raise Exception("Unhandled file format: %s" % input_file)
    audio_format = ext[1:]

    if verbose:
        print("Splitting into chunks...")
    audio_chunks = split_on_silence(sound_file, min_silence_len=min_silence_length, silence_thresh=silence_threshold)

    result = []
    name, _ = os.path.splitext(os.path.basename(input_file))
    width = len(str(len(audio_chunks)))
    format_str = "%s-%0" + str(width) + "d" + ext
    for i, chunk in enumerate(audio_chunks):
        out_file = os.path.join(output_dir, format_str % (name, i))
        if verbose:
            print("Output file: " + out_file)
        if normalize:
            chunk = effects.normalize(chunk)
        chunk.export(out_file, format=audio_format)
        result.append(out_file)

    if verbose:
        print("%d chunks generated" % len(result))

    return result


def main():
    parser = argparse.ArgumentParser(description='Splits audio files on pauses.')
    parser.add_argument('--input_file', required=True, help='The MP3/OGG/WAV file to split')
    parser.add_argument('--output_dir', required=True, help='The output directory for the splits')
    parser.add_argument('--min_silence_length', required=False, default=500, type=int, help='The minimum length in milliseconds before considering it a pause')
    parser.add_argument('--silence_threshold', required=False, default=-24, type=int, help='The threshold for considering it silent, i.e., quieter than X dBFS')
    parser.add_argument('--normalize', required=False, action='store_true', help='Whether to apply standard amplitude normalization')
    parser.add_argument('--verbose', required=False, action='store_true', help='Whether to output some debugging information')
    parsed = parser.parse_args()
    split_audio(parsed.input_file, parsed.output_dir,
                min_silence_length=parsed.min_silence_length, silence_threshold=parsed.silence_threshold,
                normalize=parsed.normalize, verbose=parsed.verbose)


if __name__ == '__main__':
    main()
