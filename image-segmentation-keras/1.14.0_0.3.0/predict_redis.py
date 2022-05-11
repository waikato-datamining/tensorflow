import io
import os
import traceback
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras_segmentation.predict import model_from_checkpoint_path
import numpy as np
import cv2
from keras_segmentation.data_utils.data_loader import get_image_array
from keras_segmentation.models.config import IMAGE_ORDERING
from PIL import Image
import random
from datetime import datetime
from rdh import Container, MessageContainer, create_parser, configure_redis, run_harness, log

# colors taken from:
# conservative 8-color palettes for color blindness
# http://mkweb.bcgsc.ca/colorblind/palettes.mhtml
class_colors = [
      0,   0,   0,
     34, 113, 178,
     61, 183, 233,
    247,  72, 165,
     53, 155, 115,
    213,  94,   0,
    230, 159,   0,
    240, 228,  66,
]
num_colors = 256 - 8
r = [random.randint(0,255) for _ in range(num_colors)]
g = [random.randint(0,255) for _ in range(num_colors)]
b = [random.randint(0,255) for _ in range(num_colors)]
for rgb in zip(r, g, b):
    class_colors.extend(rgb)


def process_image(msg_cont):
    """
    Processes the message container, loading the image from the message and forwarding the image segmentation image.

    :param msg_cont: the message container to process
    :type msg_cont: MessageContainer
    """
    config = msg_cont.params.config

    try:
        start_time = datetime.now()
        array = np.fromstring(io.BytesIO(msg_cont.message['data']).getvalue(), np.uint8)
        inp = cv2.imdecode(array, cv2.IMREAD_COLOR)

        with graph.as_default():
            if colors is not None:
                assert (len(colors) == 768), "list of colors must be 768 (256 r,g,b triplets)"
            assert len(inp.shape) == 3, "Image should be h,w,3 "

            if config.remove_background:
                inp_orig = np.copy(inp)

            output_width = model.output_width
            output_height = model.output_height
            input_width = model.input_width
            input_height = model.input_height
            n_classes = model.n_classes

            x = get_image_array(inp, input_width, input_height,
                                ordering=IMAGE_ORDERING)
            pr = model.predict(np.array([x]))[0]
            pr = pr.reshape((output_height, output_width, n_classes)).argmax(axis=2)

            original_h = inp.shape[0]
            original_w = inp.shape[1]
            pr_mask = pr.astype('uint8')
            pr_mask = cv2.resize(pr_mask, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
            if config.verbose:
                unique, count = np.unique(pr_mask, return_counts=True)
                print("  unique:", unique)
                print("  count:", count)

            if config.remove_background:
                onebit_mask = np.where(pr_mask > 0, 1, 0)
                no_background = inp_orig.copy()
                for i in range(inp_orig.shape[2]):
                    no_background[:, :, i] = no_background[:, :, i] * onebit_mask
                out_data = cv2.imencode('.png', no_background)[1].tostring()
            else:
                im = Image.fromarray(pr_mask)
                im = im.convert("P")
                if colors is not None:
                    im.putpalette(colors)
                else:
                    im.putpalette(class_colors)
                buf = io.BytesIO()
                im.save(buf, format='PNG')
                out_data = buf.getvalue()

            msg_cont.params.redis.publish(msg_cont.params.channel_out, out_data)

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


if __name__ == '__main__':
    parser = create_parser('Keras Image Segmentation - Prediction (Redis)',
                           prog="keras_seg_redis", prefix="redis_")
    parser.add_argument('--checkpoints_path', help='Directory with checkpoint file(s) and _config.json (checkpoint names: ".X" with X=0..n)', required=True, default=None)
    parser.add_argument('--memory_fraction', type=float, help='Memory fraction to use by tensorflow, i.e., limiting memory usage', required=False, default=0.5)
    parser.add_argument('--colors', help='The list of colors (RGB triplets) to use for the PNG palette, e.g.: 0,0,0,255,0,0,0,0,255 for black,red,blue', required=False, default=None)
    parser.add_argument('--remove_background', action='store_true', help='Whether to use the predicted mask to remove the background and output this modified image instead of the mask', required=False, default=False)
    parser.add_argument('--verbose', action='store_true', help='Whether to output more logging info', required=False, default=False)
    parsed = parser.parse_args()

    try:
        # apply memory usage
        print("Using memory fraction: %f" % parsed.memory_fraction)
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = parsed.memory_fraction
        set_session(tf.Session(config=config))

        # load model
        model_dir = os.path.join(parsed.checkpoints_path, '')
        print("Loading model from %s" % model_dir)
        model = model_from_checkpoint_path(model_dir)

        global graph
        graph = tf.get_default_graph()

        # color palette
        colors = []
        if parsed.colors is not None:
            colors = parsed.colors.split(",")
            colors = [int(x) for x in colors]
        if len(colors) < 768:
            colors.extend(class_colors[len(colors):])
        if len(colors) > 768:
            colors = colors[0:768]

        config = Container()
        config.model = model
        config.colors = colors
        config.remove_background = parsed.remove_background
        config.verbose = parsed.verbose

        params = configure_redis(parsed, config=config)
        run_harness(params, process_image)

    except Exception as e:
        print(traceback.format_exc())
