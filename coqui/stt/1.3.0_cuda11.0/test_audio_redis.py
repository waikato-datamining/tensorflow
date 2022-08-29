import argparse
import redis
import traceback


def test_audio(audio_file, channel, redis_host="localhost", redis_port=6379, redis_db=0):
    """
    Method for performing predictions on audio files.

    :param audio_file: the audio file to broadcast
    :type audio_file: str
    :param channel: the channel to broadcast the audio file on
    :type channel: str
    :param redis_host: the redis host to use
    :type redis_host: str
    :param redis_port: the port the redis host runs on
    :type redis_port: int
    :param redis_db: the redis database to use
    :type redis_db: int
    """

    # connect
    r = redis.Redis(host=redis_host, port=redis_port, db=redis_db)

    # prepare data
    with open(audio_file, "rb") as f:
        audio = f.read()

    r.publish(channel, audio)


def main(args=None):
    """
    Performs the test.
    Use -h to see all options.

    :param args: the command-line arguments to use, uses sys.argv if None
    :type args: list
    """
    parser = argparse.ArgumentParser(description='Coqui STT - Test audio (Redis)',
                                     prog="stt_test_audio_redis",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--redis_host', metavar='HOST', required=False, default="localhost", help='The redis server to connect to')
    parser.add_argument('--redis_port', metavar='PORT', required=False, default=6379, type=int, help='The port the redis server is listening on')
    parser.add_argument('--redis_db', metavar='DB', required=False, default=0, type=int, help='The redis database to use')
    parser.add_argument('--audio', metavar='FILE', required=True, default=None, help='The audio file to use for testing')
    parser.add_argument('--channel', metavar='ID', required=True, default=None, help='The channel to broadcast the audio data on')

    parsed = parser.parse_args(args=args)

    test_audio(parsed.image, parsed.channel,
               redis_host=parsed.redis_host, redis_port=parsed.redis_port, redis_db=parsed.redis_db)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print(traceback.format_exc())

