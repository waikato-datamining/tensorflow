import io
import traceback

from datetime import datetime
from rdh import Container, MessageContainer, create_parser, configure_redis, run_harness, log
from transcribe_utils import load_model, load_audio, transcribe_audio


def process_audio_file(msg_cont):
    """
    Processes the message container, loading the image from the message and forwarding the predictions.

    :param msg_cont: the message container to process
    :type msg_cont: MessageContainer
    """
    config = msg_cont.params.config

    try:
        start_time = datetime.now()
        if config.verbose:
            log("process_audio_files - start processing audio data")
        audio, audio_length = load_audio(io.BytesIO(msg_cont.message["data"]), config.model.sampleRate())
        pred = transcribe_audio(config.model, audio, candidate_transcripts=config.candidate_transcripts)
        msg_cont.params.redis.publish(msg_cont.params.channel_out, pred)
        if config.verbose:
            log("process_audio_files - predictions string published: %s" % msg_cont.params.channel_out)
            end_time = datetime.now()
            processing_time = end_time - start_time
            processing_time = int(processing_time.total_seconds() * 1000)
            log("process_audio_files - finished processing audio data: %d ms" % processing_time)
    except KeyboardInterrupt:
        msg_cont.params.stopped = True
    except:
        log("process_audio_files - failed to process: %s" % traceback.format_exc())


def main(args=None):
    """
    Performs the predictions.
    Use -h to see all options.

    :param args: the command-line arguments to use, uses sys.argv if None
    :type args: list
    """
    parser = create_parser('Coqui STT (Redis)', prog="stt_transcribe_redis", prefix="redis_")
    parser.add_argument('--model', metavar="FILE", type=str, required=True, help='The tflite model to use.')
    parser.add_argument("--scorer", metavar="FILE", required=False, help="Path to the external scorer file")
    parser.add_argument("--beam_width", metavar="INT", type=int, help="Beam width for the CTC decoder")
    parser.add_argument("--lm_alpha", metavar="NUM", type=float, help="Language model weight (lm_alpha). If not specified, use default from the scorer package.")
    parser.add_argument("--lm_beta", metavar="NUM", type=float, help="Word insertion bonus (lm_beta). If not specified, use default from the scorer package.")
    parser.add_argument("--candidate_transcripts", metavar="INT", type=int, default=None, help="Number of candidate transcripts to include in JSON output")
    parser.add_argument("--hot_words", metavar="WORD:BOOST[,WORD:BOOST...]", type=str, help="Hot-words and their boosts.")
    parser.add_argument('--verbose', required=False, action='store_true', help='whether to be more verbose with the output')
    parsed = parser.parse_args(args=args)

    # load model
    print("Loading model...")
    ds = load_model(parsed.model, beam_width=parsed.beam_width, scorer=parsed.scorer,
                    lm_alpha=parsed.lm_alpha, lm_beta=parsed.lm_beta, hot_words=parsed.hot_words)

    config = Container()
    config.model = ds
    config.candidate_transcripts = parsed.candidate_transcripts
    config.verbose = parsed.verbose

    params = configure_redis(parsed, config=config)
    run_harness(params, process_audio_file)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print(traceback.format_exc())
