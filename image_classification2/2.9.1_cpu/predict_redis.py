import json
import traceback

from datetime import datetime
from rdh import Container, MessageContainer, create_parser, configure_redis, run_harness, log
from predict_utils import load_labels, load_model, predict_image, read_image


def process_image(msg_cont):
    """
    Processes the message container, loading the image from the message and forwarding the predictions.

    :param msg_cont: the message container to process
    :type msg_cont: MessageContainer
    """
    config = msg_cont.params.config
    model_params = config.model_params

    try:
        start_time = datetime.now()
        if config.verbose:
            log("process_images - start processing image")
        img = read_image(msg_cont.message["data"], model_params["width"], model_params["height"])
        preds = predict_image(img, model_params)
        preds_str = json.dumps(preds, indent=2)
        msg_cont.params.redis.publish(msg_cont.params.channel_out, preds_str)
        if config.verbose:
            log("process_images - predictions string published: %s" % msg_cont.params.channel_out)
            end_time = datetime.now()
            processing_time = end_time - start_time
            processing_time = int(processing_time.total_seconds() * 1000)
            log("process_images - finished processing image: %d ms" % processing_time)
    except KeyboardInterrupt:
        msg_cont.params.stopped = True
    except:
        log("process_images - failed to process: %s" % traceback.format_exc())


def main(args=None):
    """
    Performs the predictions.
    Use -h to see all options.

    :param args: the command-line arguments to use, uses sys.argv if None
    :type args: list
    """
    parser = create_parser('tflite Image Classification Prediction (Redis)', prog="tf_predict_redis", prefix="redis_")
    parser.add_argument('--model', metavar="FILE", type=str, required=True, help='The tflite model to use.')
    parser.add_argument('--labels', metavar="FILE", type=str, required=True, help='The text file with the labels (one per line).')
    parser.add_argument('--input_mean', metavar="MEAN", type=float, required=False, default=127.5, help='The input mean to use.')
    parser.add_argument('--input_std', metavar="STD", type=float, required=False, default=127.5, help='The input standard deviation to use.')
    parser.add_argument('--num_threads', metavar="INT", type=int, required=False, default=None, help='The number of threads to use.')
    parser.add_argument("--top_x", metavar="INT", type=int, help="output only the top K labels; use <1 for all", default=5)
    parser.add_argument('--verbose', required=False, action='store_true', help='whether to be more verbose with the output')
    parsed = parser.parse_args(args=args)

    # load model
    print("Loading model...")
    model_params = load_model(parsed.model, num_threads=parsed.num_threads)
    model_params["input_mean"] = parsed.input_mean
    model_params["input_std"] = parsed.input_std
    model_params["top_x"] = parsed.top_x
    model_params["labels"] = load_labels(parsed.labels)

    config = Container()
    config.model_params = model_params
    config.verbose = parsed.verbose

    params = configure_redis(parsed, config=config)
    run_harness(params, process_image)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print(traceback.format_exc())
