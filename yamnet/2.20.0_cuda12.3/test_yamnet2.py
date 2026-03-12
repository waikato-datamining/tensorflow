# based on:
# https://github.com/tensorflow/docs/blob/master/site/en/hub/tutorials/yamnet.ipynb

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import csv
import sys

from scipy.io import wavfile

# Load the model.
model = hub.load('https://tfhub.dev/google/yamnet/1')



# Find the name of the class with the top score when mean-aggregated across frames.
def class_names_from_csv(class_map_csv_text):
  """Returns list of class names corresponding to score vector."""
  class_names = []
  with tf.io.gfile.GFile(class_map_csv_text) as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
      class_names.append(row['display_name'])

  return class_names

class_map_path = model.class_map_path().numpy()
class_names = class_names_from_csv(class_map_path)
     


def ensure_sample_rate(original_sample_rate, waveform,
                       desired_sample_rate=16000):
  """Resample waveform if required."""
  if original_sample_rate != desired_sample_rate:
    desired_length = int(round(float(len(waveform)) /
                               original_sample_rate * desired_sample_rate))
    waveform = scipy.signal.resample(waveform, desired_length)
  return desired_sample_rate, waveform
     
# wav_file_name = 'speech_whistling2.wav'
if len(sys.argv) > 1:
  wav_file_name = sys.argv[1]
else:
  wav_file_name = 'miaow_16k.wav'
print("Using wav file: %s" % wav_file_name)
sample_rate, wav_data = wavfile.read(wav_file_name, 'rb')
sample_rate, wav_data = ensure_sample_rate(sample_rate, wav_data)

# Show some basic information about the audio.
duration = len(wav_data)/sample_rate
print(f'Sample rate: {sample_rate} Hz')
print(f'Total duration: {duration:.2f}s')
print(f'Size of the input: {len(wav_data)}')

waveform = wav_data / tf.int16.max
     
# Run the model, check the output.
scores, embeddings, spectrogram = model(waveform)
     
scores_np = scores.numpy()
spectrogram_np = spectrogram.numpy()
infered_class = class_names[scores_np.mean(axis=0).argmax()]
print(f'The main sound is: {infered_class}')
     

