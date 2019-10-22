"""
Exposes the main() function as an executable.
"""
import sys
import traceback

from wai.tfrecords.adams2objectdetection import main

try:
    main(sys.argv[1:])
except Exception as ex:
    print(traceback.format_exc())
