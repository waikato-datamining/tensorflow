import io
import traceback

from datetime import datetime
from wai.tmm.objdet.predict_utils import load_model, load_classes, preprocess_image_bytes, detect_objects
from rdh import Container, MessageContainer, create_parser, configure_redis, run_harness, log


def process_image(msg_cont):
    """
    Processes the message container, loading the image from the message and forwarding the object detection predictions.

    :param msg_cont: the message container to process
    :type msg_cont: MessageContainer
    """
    config = msg_cont.params.config

    try:
        start_time = datetime.now()
        _, input_height, input_width, _ = config.model.get_input_details()[0]['shape']

        image = io.BytesIO(msg_cont.message['data']).getvalue()
        preprocessed_image, original_image = preprocess_image_bytes(image, (input_height, input_width))
        image_height, image_width, _ = original_image.shape
        results = detect_objects(config.model, preprocessed_image, (image_height, image_width), threshold=config.threshold, labels=config.labels)

        msg_cont.params.redis.publish(msg_cont.params.channel_out, results.to_json_string())

        if config.verbose:
            log("process_images - predicted image published: %s" % msg_cont.params.channel_out)
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
    The main method for parsing command-line arguments and starting the training.

    :param args: the commandline arguments, uses sys.argv if not supplied
    :type args: list
    """

    parser = create_parser("Uses a tflite object detection model to make predictions on images received via Redis and sends predictions back to Redis.",
                           prog="tmm-od-predict-redis", prefix="redis_")
    parser.add_argument('--model', metavar="FILE", type=str, required=True, help='The tflite object detection model to use.')
    parser.add_argument('--labels', metavar="FILE", type=str, required=True, help='The text file with the labels (one label per line).')
    parser.add_argument('--threshold', metavar="0-1", type=float, required=False, default=0.3, help='The probability threshold to use.')
    parser.add_argument('--verbose', action="store_true", required=False, help='Whether to output debugging information.')
    parsed = parser.parse_args(args=args)

    log("Loading model: %s" % parsed.model)
    model = load_model(parsed.model)

    log("Loading labels: %s" % parsed.labels)
    labels = load_classes(parsed.labels)

    config = Container()
    config.model = model
    config.labels = labels
    config.threshold = parsed.threshold
    config.verbose = parsed.verbose

    params = configure_redis(parsed, config=config)
    run_harness(params, process_image)


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
