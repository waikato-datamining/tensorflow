import argparse
import os
import traceback
from sfp import Poller
from transcribe_utils import load_model, load_audio, transcribe_audio


SUPPORTED_EXTS = [".wav"]
""" supported file extensions (lower case with dot). """


def process_audio(fname, output_dir, poller):
    """
    Method for processing an audio file.

    :param fname: the audio file to process
    :type fname: str
    :param output_dir: the directory to write the audio file to
    :type output_dir: str
    :param poller: the Poller instance that called the method
    :type poller: Poller
    :return: the list of generated output files
    :rtype: list
    """
    result = []
    try:
        if poller.params.candidate_transcripts is not None:
            out_path = "{}/{}{}".format(output_dir, os.path.splitext(os.path.basename(fname))[0], ".json")
        else:
            out_path = "{}/{}{}".format(output_dir, os.path.splitext(os.path.basename(fname))[0], ".txt")
        audio, audio_length = load_audio(fname, poller.params.model.sampleRate())
        pred = transcribe_audio(poller.params.model, audio, candidate_transcripts=poller.params.candidate_transcripts)
        with open(out_path, "w") as fp:
            fp.write(pred)
        result.append(out_path)
    except KeyboardInterrupt:
        poller.keyboard_interrupt()
    except:
        poller.error("Failed to process audio file: %s\n%s" % (fname, traceback.format_exc()))
    return result


def transcribe_audio_files(model, input_dir, output_dir, tmp_dir=None,
                           poll_wait=1.0, continuous=False, use_watchdog=False, watchdog_check_interval=10.0,
                           delete_input=False, beam_width=None, scorer=None, lm_alpha=None, lm_beta=None,
                           hot_words=None, candidate_transcripts=None, verbose=False, quiet=False):
    """
    Generates transcriptions for audio files found in input_dir and outputs the prediction TXT/JSON files in output_dir.

    :param model: the tflite model file to use
    :param model: str
    :param input_dir: the directory with the audio files
    :type input_dir: str
    :param output_dir: the output directory to move the audio files to and store the predictions
    :type output_dir: str
    :param tmp_dir: the temporary directory to store the predictions until finished
    :type tmp_dir: str
    :param poll_wait: the amount of seconds between polls when not in watchdog mode
    :type poll_wait: float
    :param continuous: whether to poll for files continuously
    :type continuous: bool
    :param use_watchdog: whether to react to file creation events rather than use fixed-interval polling
    :type use_watchdog: bool
    :param watchdog_check_interval: the interval for the watchdog process to check for files that were missed due to potential race conditions
    :type watchdog_check_interval: float
    :param delete_input: whether to delete the input audio files rather than moving them to the output directory
    :type delete_input: bool
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
    :param candidate_transcripts: the number of candidate transcripts to return, use None to return text transcript instead of json
    :type candidate_transcripts: int
    :param verbose: whether to output more logging information
    :type verbose: bool
    :param quiet: whether to suppress output
    :type quiet: bool
    """

    if verbose:
        print("Loading model: %s" % model)
    ds = load_model(model, beam_width=beam_width, scorer=scorer, lm_alpha=lm_alpha, lm_beta=lm_beta, hot_words=hot_words)

    poller = Poller()
    poller.input_dir = input_dir
    poller.output_dir = output_dir
    poller.tmp_dir = tmp_dir
    poller.extensions = SUPPORTED_EXTS
    poller.delete_input = delete_input
    poller.verbose = verbose
    poller.progress = not quiet
    poller.check_file = None
    poller.process_file = process_audio
    poller.poll_wait = poll_wait
    poller.continuous = continuous
    poller.use_watchdog = use_watchdog
    poller.watchdog_check_interval = watchdog_check_interval
    poller.params.model = ds
    poller.params.candidate_transcripts = candidate_transcripts
    poller.poll()


def main(args=None):
    """
    The main method for parsing command-line arguments and starting the training.

    :param args: the commandline arguments, uses sys.argv if not supplied
    :type args: list
    """

    parser = argparse.ArgumentParser(
        description="Coqui STT (file-polling)",
        prog="stt_transcribe_poll",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', metavar="FILE", type=str, required=True, help='The tflite model to use.')
    parser.add_argument('--prediction_in', help='Path to the input audio files', required=True, default=None)
    parser.add_argument('--prediction_out', help='Path to the output files folder', required=True, default=None)
    parser.add_argument('--prediction_tmp', help='Path to the temporary files folder', required=False, default=None)
    parser.add_argument('--poll_wait', type=float, help='poll interval in seconds when not using watchdog mode', required=False, default=1.0)
    parser.add_argument('--continuous', action='store_true', help='Whether to continuously process audio files and perform prediction', required=False, default=False)
    parser.add_argument('--use_watchdog', action='store_true', help='Whether to react to file creation events rather than performing fixed-interval polling', required=False, default=False)
    parser.add_argument('--watchdog_check_interval', type=float, help='check interval in seconds for the watchdog', required=False, default=10.0)
    parser.add_argument('--delete_input', action='store_true', help='Whether to delete the input files rather than move them to --prediction_out directory', required=False, default=False)
    parser.add_argument("--scorer", metavar="FILE", required=False, help="Path to the external scorer file")
    parser.add_argument("--beam_width", metavar="INT", type=int, help="Beam width for the CTC decoder")
    parser.add_argument("--lm_alpha", metavar="NUM", type=float, help="Language model weight (lm_alpha). If not specified, use default from the scorer package.")
    parser.add_argument("--lm_beta", metavar="NUM", type=float, help="Word insertion bonus (lm_beta). If not specified, use default from the scorer package.")
    parser.add_argument("--candidate_transcripts", metavar="INT", type=int, default=None, help="Number of candidate transcripts to include in JSON output")
    parser.add_argument("--hot_words", metavar="WORD:BOOST[,WORD:BOOST...]", type=str, help="Hot-words and their boosts.")
    parser.add_argument('--verbose', action='store_true', help='Whether to output more logging info', required=False, default=False)
    parser.add_argument('--quiet', action='store_true', help='Whether to suppress output', required=False, default=False)
    parsed = parser.parse_args(args=args)

    transcribe_audio_files(parsed.model, parsed.prediction_in, parsed.prediction_out, tmp_dir=parsed.prediction_tmp,
                           poll_wait=parsed.poll_wait, continuous=parsed.continuous, use_watchdog=parsed.use_watchdog,
                           watchdog_check_interval=parsed.watchdog_check_interval, delete_input=parsed.delete_input,
                           scorer=parsed.scorer, beam_width=parsed.beam_width, lm_alpha=parsed.lm_alpha,
                           lm_beta=parsed.lm_beta, candidate_transcripts=parsed.candidate_transcripts,
                           hot_words=parsed.hot_words, verbose=parsed.verbose, quiet=parsed.quiet)


def sys_main() -> int:
    """
    Runs the main function using the system cli arguments, and
    returns a system error code.
    :return:    0 for success, 1 for failure.
    """
    try:
        main()
        return 0
    except Exception:
        print(traceback.format_exc())
        return 1


if __name__ == '__main__':
    main()
