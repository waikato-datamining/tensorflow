# Original code from:
# https://stt.readthedocs.io/en/latest/_downloads/67bac4343abf2261d69231fdaead59fb/client.py

import json
import shlex
import subprocess
import sys
import wave

import numpy as np

from stt import Model, version
from timeit import default_timer as timer

try:
    from shlex import quote
except ImportError:
    from pipes import quote


def load_model(model, beam_width=None, scorer=None, lm_alpha=None, lm_beta=None, hot_words=None, verbose=False):
    """
    Loads the specified model from disk.

    :param model: the filename of the tflite model to load
    :type model: str
    :param beam_width: the beam width for the CTC decoder, ignored if None
    :type beam_width: int
    :param scorer: the filename of the external scorer to load, ignored if None
    :type scorer: str
    :param lm_alpha: the language model weight (lm_alpha). If None, use default from the scorer package.
    :type lm_alpha: float
    :param lm_beta: the word insertion bonus (lm_beta). If None, use default from the scorer package.
    :type lm_beta: float
    :param hot_words: Hot-words and their boosts (comma-separated list of WORD:BOOST).
    :type hot_words: str
    :param verbose: whether to output some debugging information
    :type verbose: bool
    :return: the model
    """

    if verbose:
        print("Loading model from file {}".format(model), file=sys.stderr)
    model_load_start = timer()
    # sphinx-doc: python_ref_model_start
    ds = Model(model)
    # sphinx-doc: python_ref_model_stop
    model_load_end = timer() - model_load_start
    if verbose:
        print("Loaded model in {:.3}s.".format(model_load_end), file=sys.stderr)

    if beam_width is not None:
        ds.setBeamWidth(beam_width)

    if scorer is not None:
        print("Loading scorer from files {}".format(scorer), file=sys.stderr)
        scorer_load_start = timer()
        ds.enableExternalScorer(scorer)
        scorer_load_end = timer() - scorer_load_start
        if verbose:
            print("Loaded scorer in {:.3}s.".format(scorer_load_end), file=sys.stderr)

        if (lm_alpha is not None) and (lm_beta is not None):
            ds.setScorerAlphaBeta(lm_alpha, lm_beta)

    if hot_words is not None:
        if verbose:
            print("Adding hot-words", file=sys.stderr)
        for word_boost in hot_words.split(","):
            word, boost = word_boost.split(":")
            ds.addHotWord(word, float(boost))

    return ds


def convert_samplerate(audio_path, desired_sample_rate):
    """
    Converts a WAV file to the specified sample rate.

    :param audio_path: the WAV file to convert
    :type audio_path: str
    :param desired_sample_rate: the target sample rate
    :type desired_sample_rate: int
    :return: tuple of desired sample rate and audio file as numpy array
    :rtype: tuple
    """
    sox_cmd = "sox {} --type raw --bits 16 --channels 1 --rate {} --encoding signed-integer --endian little --compression 0.0 --no-dither - ".format(
        quote(audio_path), desired_sample_rate
    )
    try:
        output = subprocess.check_output(shlex.split(sox_cmd), stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        raise RuntimeError("SoX returned non-zero status: {}".format(e.stderr))
    except OSError as e:
        raise OSError(
            e.errno,
            "SoX not found, use {}hz files or install it: {}".format(
                desired_sample_rate, e.strerror
            ),
        )

    return desired_sample_rate, np.frombuffer(output, np.int16)


def load_audio(audio, desired_sample_rate):
    """
    Loads the audio and resamples it if necessary.

    :param audio: the file or fp to read the audio from
    :param desired_sample_rate: the target sample rate
    :type desired_sample_rate: int
    :return: the tuple of audio numpy array and audio length
    :rtype: tuple
    """

    fin = wave.open(audio, "rb")
    fs_orig = fin.getframerate()
    if fs_orig != desired_sample_rate:
        print(
            "Warning: original sample rate ({}) is different than {}hz. Resampling might produce erratic speech recognition.".format(
                fs_orig, desired_sample_rate
            ),
            file=sys.stderr,
        )
        fs_new, audio = convert_samplerate(audio, desired_sample_rate)
    else:
        audio = np.frombuffer(fin.readframes(fin.getnframes()), np.int16)

    audio_length = fin.getnframes() * (1 / fs_orig)
    fin.close()

    return audio, audio_length


def metadata_to_string(metadata):
    """
    Returns the meta-data as string.

    :param metadata: the meta-data to convert
    :return: the generated string
    :rtype: str
    """
    return "".join(token.text for token in metadata.tokens)


def words_from_candidate_transcript(metadata):
    """
    Extracts the words from the candidate transcript.

    :param metadata: the meta-data to extract the words from
    :return: the word list
    :rtype: list
    """
    word = ""
    word_list = []
    word_start_time = 0
    # Loop through each character
    for i, token in enumerate(metadata.tokens):
        # Append character to word if it's not a space
        if token.text != " ":
            if len(word) == 0:
                # Log the start time of the new word
                word_start_time = token.start_time

            word = word + token.text
        # Word boundary is either a space or the last character in the array
        if token.text == " " or i == len(metadata.tokens) - 1:
            word_duration = token.start_time - word_start_time

            if word_duration < 0:
                word_duration = 0

            each_word = dict()
            each_word["word"] = word
            each_word["start_time"] = round(word_start_time, 4)
            each_word["duration"] = round(word_duration, 4)

            word_list.append(each_word)
            # Reset
            word = ""
            word_start_time = 0

    return word_list


def metadata_json_output(metadata):
    """
    Turns the meta-data into json format.

    :param metadata: the meta-data to generate json from
    :return: the generated JSON string
    :rtype: str
    """
    json_result = dict()
    json_result["transcripts"] = [
        {
            "confidence": transcript.confidence,
            "words": words_from_candidate_transcript(transcript),
        }
        for transcript in metadata.transcripts
    ]
    return json.dumps(json_result, indent=2)


def transcribe_audio(ds, audio, candidate_transcripts=None):
    """
    Transcribes the audio clip and returns the result, either simple text
    or a json string for the candidate transcripts (if not None).

    :param ds: the model to use
    :param audio: the audio clip as numpy array (with the correct sample rate)
    :param candidate_transcripts: the number of candidate transcripts to return, use None to just return text
    :type candidate_transcripts: int
    :return: the transcript of the JSON with the candidate transcripts
    :rtype: str
    """
    if candidate_transcripts is None:
        return ds.stt(audio)
    else:
        return metadata_json_output(ds.sttWithMetadata(audio, candidate_transcripts))
