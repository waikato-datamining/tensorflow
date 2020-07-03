#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from datetime import datetime
from deepspeech import Model
from deepspeech.client import convert_samplerate, metadata_json_output, metadata_to_string
import numpy as np
import time
import traceback
import os
from pydub import AudioSegment
import sys
import wave

SUPPORTED_EXTS = [".mp3", ".ogg", ".wav"]
""" supported file extensions (lower case). """


def process(model, prediction_in, prediction_out, prediction_tmp, continuous, delete_input,
            json, candidate_transcripts):
    """
    Process sound files.

    :param model: the model to use for inference
    :type model: Model
    :param prediction_in: the input directory with sound files
    :type prediction_in: str
    :param prediction_out: the output directory for processed sound files and model output
    :type prediction_out: str
    :param prediction_tmp: the directory where to store the model output initially before moving it into prediction_out
    :type prediction_tmp: str
    :param continuous: whether to run in continuous mode
    :type continuous: bool
    :param delete_input: if True sound files get deleted rather than moved to prediction_out
    :type delete_input: bool
    :param json: whether to output model output in JSON rather just the text
    :type json: bool
    :param candidate_transcripts: the number of candidate transcripts to include in JSON output
    :type candidate_transcripts: int
    """

    desired_sample_rate = model.sampleRate()

    while True:
        start_time = datetime.now()
        sound_files = []
        for file_path in os.listdir(prediction_in):
            # Process sound files only
            ext_lower = os.path.splitext(file_path)[1].lower()
            if ext_lower in SUPPORTED_EXTS:
                full_path = os.path.join(prediction_in, file_path)
                sound_files.append(full_path)
        if len(sound_files) == 0:
            if continuous:
                time.sleep(1)
                continue
            else:
                return

        for sound_file in sound_files:
            print("%s - %s" % (str(datetime.now()), os.path.basename(sound_file)))
            try:
                # convert to WAV?
                if sound_file.lower().endswith(".mp3"):
                    tmp_file = os.path.join(prediction_out, os.path.basename(sound_file) + ".wav")
                    audio = AudioSegment.from_mp3(sound_file)
                    audio.export(tmp_file, format="wav")
                elif sound_file.lower().endswith(".ogg"):
                    tmp_file = os.path.join(prediction_out, os.path.basename(sound_file) + ".wav")
                    audio = AudioSegment.from_ogg(sound_file)
                    audio.export(tmp_file, format="wav")
                elif sound_file.lower().endswith(".wav"):
                    tmp_file = None
                else:
                    print("  unsupported file format: %s" % sound_file)
                    continue

                # load audio file
                if tmp_file is None:
                    fin = wave.open(sound_file, 'rb')
                else:
                    fin = wave.open(tmp_file, 'rb')
                fs_orig = fin.getframerate()
                if fs_orig != desired_sample_rate:
                    print('  Warning: original sample rate ({}) is different than {}hz.\n  Resampling might produce erratic speech recognition.'.format(fs_orig, desired_sample_rate), file=sys.stderr)
                    fs_new, audio = convert_samplerate(sound_file, desired_sample_rate)
                else:
                    audio = np.frombuffer(fin.readframes(fin.getnframes()), np.int16)
                fin.close()

                # perform inference
                if prediction_tmp is None:
                    out_dir = prediction_out
                else:
                    out_dir = prediction_tmp
                if json:
                    out_file = os.path.join(out_dir, os.path.splitext(os.path.basename(sound_file))[0] + ".json")
                    content = metadata_json_output(model.sttWithMetadata(audio, candidate_transcripts))
                else:
                    out_file = os.path.join(out_dir, os.path.splitext(os.path.basename(sound_file))[0] + ".txt")
                    content = model.stt(audio)
                with open(out_file, 'w') as of:
                    of.write(content)
                    of.write("\n")
                if prediction_tmp is not None:
                    os.rename(out_file, os.path.join(prediction_out, os.path.basename(out_file)))
                if tmp_file is not None:
                    os.remove(tmp_file)

            except:
                print("Failed to process: %s" % sound_file)
                print(traceback.format_exc())

            if delete_input:
                os.remove(sound_file)
            else:
                os.rename(sound_file, os.path.join(prediction_out, os.path.basename(sound_file)))

            end_time = datetime.now()
            inference_time = end_time - start_time
            print("  Inference + I/O time: {} ms\n".format(inference_time))


def main():
    parser = argparse.ArgumentParser(description='Running DeepSpeech inference.')
    parser.add_argument('--model', required=True, help='Path to the model (protocol buffer binary file)')
    parser.add_argument('--scorer', required=False, help='Path to the external scorer file')
    parser.add_argument('--prediction_in', required=True, help='Path to the directory with sound files (mp3/wav) to analyze')
    parser.add_argument('--prediction_out', required=True, help='Path to the directory for moving the processed sound files to')
    parser.add_argument('--prediction_tmp', required=False, help='Path to the temp directory for storing the predictions initially before moving them to "--prediction_out"')
    parser.add_argument('--continuous', action='store_true', help='Whether to continuously load test images and perform prediction', required=False, default=False)
    parser.add_argument('--delete_input', action='store_true', help='Whether to delete the input files rather than move them to "--prediction_out" directory', required=False, default=False)
    parser.add_argument('--beam_width', type=int, help='Beam width for the CTC decoder')
    parser.add_argument('--lm_alpha', type=float, help='Language model weight (lm_alpha). If not specified, use default from the scorer package.')
    parser.add_argument('--lm_beta', type=float, help='Word insertion bonus (lm_beta). If not specified, use default from the scorer package.')
    parser.add_argument('--json', required=False, action='store_true', help='Output json from metadata with timestamp of each word')
    parser.add_argument('--candidate_transcripts', type=int, default=3, help='Number of candidate transcripts to include in JSON output')
    parsed = parser.parse_args()

    print('Loading model from file {}'.format(parsed.model))
    ds = Model(parsed.model)
    if parsed.beam_width:
        ds.setBeamWidth(parsed.beam_width)

    if parsed.scorer:
        print('Loading scorer from file {}'.format(parsed.scorer))
        ds.enableExternalScorer(parsed.scorer)
        if parsed.lm_alpha and parsed.lm_beta:
            ds.setScorerAlphaBeta(parsed.lm_alpha, parsed.lm_beta)

    process(model=ds,
            prediction_in=parsed.prediction_in, prediction_out=parsed.prediction_out, prediction_tmp=parsed.prediction_tmp,
            continuous=parsed.continuous, delete_input=parsed.delete_input,
            json=parsed.json, candidate_transcripts=parsed.candidate_transcripts)


if __name__ == '__main__':
    main()
