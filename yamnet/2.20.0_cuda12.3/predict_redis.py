import io
import json
import traceback
from datetime import datetime

from rdh import Container, MessageContainer, create_parser, configure_redis, run_harness, log

from predict_common import load_model, load_audio, class_names_from_model, predict


def process_audio(msg_cont):
    """
    Processes the message container, loading the audio from the message and forwarding the predictions.

    :param msg_cont: the message container to process
    :type msg_cont: MessageContainer
    """
    config = msg_cont.params.config

    try:
        start_time = datetime.now()
        if config.verbose:
            log("process_audio - start processing audio")
            
        wav = load_audio(io.BytesIO(msg_cont.message['data']))
        preds = predict(config.model, wav, config.class_names)
        preds_str = json.dumps(preds, indent=2)
        msg_cont.params.redis.publish(msg_cont.params.channel_out, preds_str)
        if config.verbose:
            log("process_audio - predictions string published: %s" % msg_cont.params.channel_out)
            end_time = datetime.now()
            processing_time = end_time - start_time
            processing_time = int(processing_time.total_seconds() * 1000)
            log("process_audio - finished processing audio: %d ms" % processing_time)
    except KeyboardInterrupt:
        msg_cont.params.stopped = True
    except:
        log("process_audio - failed to process: %s" % traceback.format_exc())


def main(args=None):
    """
    Performs the predictions.
    Use -h to see all options.

    :param args: the command-line arguments to use, uses sys.argv if None
    :type args: list
    """
    parser = create_parser('yamnet - Prediction (Redis)', prog="yamnet_predict_redis", prefix="redis_")
    parser.add_argument('--verbose', required=False, action='store_true', help='whether to be more verbose with the output')

    parsed = parser.parse_args(args=args)

    # load model
    print("Loading model...")
    model = load_model()

    config = Container()
    config.model = model
    config.class_names = class_names_from_model(model)
    config.verbose = parsed.verbose

    params = configure_redis(parsed, config=config)
    run_harness(params, process_audio)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print(traceback.format_exc())
