from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import config as cfg
import download_images as download

import tensorflow as tf
import numpy as np

from datetime import datetime
import os
import random
import sys
import threading

tf.app.flags.DEFINE_string('train_directory', 'train-' + str(cfg.dataset['training_set_size']) + '-test-' + str(cfg.dataset['testing_set_size']) + '/' + cfg.directory['training'],
                           'Training data directory')
tf.app.flags.DEFINE_string('validation_directory', 'train-' + str(cfg.dataset['training_set_size']) + '-test-' + str(cfg.dataset['testing_set_size']) + '/' + cfg.directory['testing'],
                           'Validation data directory')
tf.app.flags.DEFINE_string('output_directory', 'train-' + str(cfg.dataset['training_set_size']) + '-test-' + str(cfg.dataset['testing_set_size']) + '/' + cfg.directory['output'],
                           'Output data directory')
                         
tf.app.flags.DEFINE_integer('train_shards', cfg.shards['training'],
                            'Number of shards in training TFRecord files.')
tf.app.flags.DEFINE_integer('validation_shards', cfg.shards['testing'],
                            'Number of shards in validation TFRecord files.')
                            
tf.app.flags.DEFINE_integer('num_threads', cfg.threads['preprocess'],
                            'Number of threads to preprocess the images.')

tf.app.flags.DEFINE_integer('noise_scale_percentage', cfg.image['noise_scale'], 
							'Standard deviation of the normal noise introduced (as a percentage).')

FLAGS = tf.app.flags.FLAGS

print(cfg.image['height'])
print(FLAGS.train_directory)

######################################################
# Wrappers for inserting features into Example protos.
######################################################
def _int64_feature(value):
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
  
def _float_feature(value):
  if not isinstance(value, list):
  	value = [value]
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))
  
######################################################
# Converting to Example proto
######################################################
def _convert_to_example(filename, image_buffer, label, text, height, width, avg_intensity, intensity_label):

  example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': _int64_feature(height),
      'image/width': _int64_feature(width),
      'image/colorspace': _bytes_feature(tf.compat.as_bytes(cfg.image['colorspace'])),
      'image/channels': _int64_feature(cfg.image['channels']),
      'image/class/label': _int64_feature(label),
      'image/class/text': _bytes_feature(tf.compat.as_bytes(text)),
      'image/format': _bytes_feature(tf.compat.as_bytes(cfg.image['format'])),
      'image/filename': _bytes_feature(tf.compat.as_bytes(os.path.basename(filename))),
      'image/encoded': _bytes_feature(tf.compat.as_bytes(image_buffer)),
      'image/avg_intensity': _float_feature(avg_intensity),
      'image/intensity_label': _float_feature(intensity_label)}))
      
  return example

######################################################
# Image coder class
# Helper class that provides TensorFlow image coding utilities.
######################################################

class ImageCoder(object):

  def __init__(self):
    # Create a single Session to run all image coding calls.
    self._sess = tf.Session()

    # Initializes function that converts PNG to JPEG data.
    self._png_data = tf.placeholder(dtype=tf.string)
    image = tf.image.decode_png(self._png_data, channels=3)
    self._png_to_jpeg = tf.image.encode_jpeg(image, format='rgb', quality=100)

    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg_tmp = tf.image.decode_jpeg(self._decode_jpeg_data, channels=cfg.image['channels'])
    self._decode_jpeg = tf.image.resize_images(self._decode_jpeg_tmp, [cfg.image['height'], cfg.image['width']])

  def png_to_jpeg(self, image_data):
    return self._sess.run(self._png_to_jpeg,
                          feed_dict={self._png_data: image_data})

  def decode_jpeg(self, image_data):
    image = self._sess.run(self._decode_jpeg,
                           feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == cfg.image['channels']
    return image

######################################################
# Determine if a file contains a PNG format image.
#  Args:
#    filename: string, path of the image file.
#  Returns:
#    boolean indicating if the image is a PNG.
######################################################
def _is_png(filename):
  return '.png' in filename

######################################################
# Process a single image file
#  Args:
#    filename: string, path of the image file.
#    coder: instance of ImageCoder to provide TensorFlow image coding utils.
#  Returns:
#    image_buffer: string, JPEG encoding of RGB image.
#    height: integer, image height in pixels.
#    width: integer, image width in pixels.
#    avg_intensity: float, average pixel intensity in image [0, 255]
#    intensity_label: float, calculated average intensity loaded with normal noise [0, 255]
######################################################
def _process_image(filename, coder):

  # Read the image file.
  with tf.gfile.FastGFile(filename, 'r') as f:
    image_data = f.read()

  # Convert any PNG to JPEG's for consistency.
  if _is_png(filename):
    print('Converting PNG to JPEG for %s' % filename)
    image_data = coder.png_to_jpeg(image_data)
  
  # Decode the RGB JPEG.
  image = coder.decode_jpeg(image_data)

  # Check that image converted to grayscale
  assert len(image.shape) == 3
  height = image.shape[0]
  width = image.shape[1]
  assert image.shape[2] == cfg.image['channels']
  
  image = image.ravel()
  
  # Get average pixel intensity
  length = len(image)
  sum_intensity = 0
  image_intensity = image.astype(np.float)
  for i in image_intensity:
  	sum_intensity += i
  avg_intensity = sum_intensity.astype(np.float) / length
  
  noise = np.random.normal(scale=FLAGS.noise_scale_percentage / 100 * 255 )
  
  intensity_label = avg_intensity + noise
  
  if intensity_label > 255:
  	intensity_label = 255
  if intensity_label < 0:
  	intensity_label = 0
  	
  image=image.astype(np.uint8).tostring()

  return image, height, width, avg_intensity, intensity_label

######################################################
# Process and save a list of images as TFRecord in 1 thread.
# Args:
#    coder: instance of ImageCoder to provide TensorFlow image coding utils.
#    thread_index: integer, unique batch to run index is within [0, len(ranges)).
#    ranges: list of pairs of integers specifying ranges of each batches to
#      analyze in parallel.
#    name: string, unique identifier specifying the data set
#    filenames: list of strings; each string is a path to an image file
#    texts: list of strings; each string is human readable, e.g. 'dog'
#    labels: list of integer; each integer identifies the ground truth
#    num_shards: integer number of shards for this data set.
######################################################
def _process_image_files_batch(coder, thread_index, ranges, name, filenames,
                               texts, labels, num_shards):
  
  # Each thread produces N shards where N = int(num_shards / num_threads).
  num_threads = len(ranges)
  assert not num_shards % num_threads
  num_shards_per_batch = int(num_shards / num_threads)

  shard_ranges = np.linspace(ranges[thread_index][0],
                             ranges[thread_index][1],
                             num_shards_per_batch + 1).astype(int)
  num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

  counter = 0
  for s in range(num_shards_per_batch):
    # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
    shard = thread_index * num_shards_per_batch + s
    output_filename = '%s-%s-%.5d-of-%.5d' % (cfg.tfRecords['prefix'], name, shard, num_shards)
    print('output filename')
    print(output_filename)
    
    output_file = os.path.join(FLAGS.output_directory, output_filename)
    print(output_file)
    writer = tf.python_io.TFRecordWriter(output_file)
    
    """if not os.path.exists(FLAGS.output_directory):
    	os.makedirs(FLAGS.output_directory)"""

    shard_counter = 0
    files_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
    for i in files_in_shard:
      filename = filenames[i]
      label = labels[i]
      text = texts[i]

      image_buffer, height, width, avg_intensity, intensity_label = _process_image(filename, coder)

      example = _convert_to_example(filename, image_buffer, label,
                                    text, height, width, avg_intensity, intensity_label)
      writer.write(example.SerializeToString())
      shard_counter += 1
      counter += 1

      if not counter % 1000:
        print('%s [thread %d]: Processed %d of %d images in thread batch.' %
              (datetime.now(), thread_index, counter, num_files_in_thread))
        sys.stdout.flush()

    writer.close()
    print('%s [thread %d]: Wrote %d images to %s' %
          (datetime.now(), thread_index, shard_counter, output_file))
    sys.stdout.flush()
    shard_counter = 0
  print('%s [thread %d]: Wrote %d images to %d shards.' %
        (datetime.now(), thread_index, counter, num_files_in_thread))
  sys.stdout.flush()

######################################################
# Process and save a list of images as TFRecord of Example protos
#  Args:
#    name: string, unique identifier specifying the data set
#    filenames: list of strings; each string is a path to an image file
#    texts: list of strings; each string is human readable, e.g. 'dog'
#    labels: list of integer; each integer identifies the ground truth
#    num_shards: integer number of shards for this data set.
######################################################
def _process_image_files(name, filenames, texts, labels, num_shards):

  assert len(filenames) == len(texts)
  assert len(filenames) == len(labels)

  # Break all images into batches with a [ranges[i][0], ranges[i][1]].
  spacing = np.linspace(0, len(filenames), FLAGS.num_threads + 1).astype(np.int)
  ranges = []
  for i in range(len(spacing) - 1):
    ranges.append([spacing[i], spacing[i+1]])

  # Launch a thread for each batch.
  print('Launching %d threads for spacings: %s' % (FLAGS.num_threads, ranges))
  sys.stdout.flush()

  # Create a mechanism for monitoring when all threads are finished.
  coord = tf.train.Coordinator()

  # Create a generic TensorFlow-based utility for converting all image codings.
  coder = ImageCoder()

  threads = []
  for thread_index in range(len(ranges)):
    args = (coder, thread_index, ranges, name, filenames,
            texts, labels, num_shards)
    t = threading.Thread(target=_process_image_files_batch, args=args)
    t.start()
    threads.append(t)

  # Wait for all the threads to terminate.
  coord.join(threads)
  print('%s: Finished writing all %d images in data set.' %
        (datetime.now(), len(filenames)))
  sys.stdout.flush()

######################################################
# Build a list of all images files and labels in the data set.
#   Args:
#    data_dir: string, path to the root directory of images.
#  Returns:
#    filenames: list of strings; each string is a path to an image file.
#    texts: list of strings; each string is the class, e.g. 'dog'
#    labels: list of integer; each integer identifies the ground truth.
######################################################
def _find_image_files(data_dir):

  print('Determining list of input files and labels from %s.' % data_dir)

  labels = []
  filenames = []
  texts = []

  # Leave label index 0 empty as a background class.
  label_index = 1
    
  # Construct the list of JPEG files and labels.
  for i in range(cfg.dataset['num_categories']):
    jpeg_file_path = '%s/%s/*' % (data_dir, cfg.dataset['label' + str(i) + '_folder'])
    matching_files = tf.gfile.Glob(jpeg_file_path)

    labels.extend([label_index] * len(matching_files))
    texts.extend([cfg.dataset['label' + str(i) + '_folder']] * len(matching_files))
    filenames.extend(matching_files)

    if not label_index % 100:
      print('Finished finding files in %d of %d classes.' % (
          label_index, len(labels)))
    label_index += 1

  # Shuffle the ordering of all image files to guarantee random ordering of the
  # images with respect to label in the saved TFRecord files.
  shuffled_index = list(range(len(filenames)))
  random.seed(12345)
  random.shuffle(shuffled_index)

  filenames = [filenames[i] for i in shuffled_index]
  texts = [texts[i] for i in shuffled_index]
  labels = [labels[i] for i in shuffled_index]

  print('Found %d JPEG files across %d labels inside %s.' %
        (len(filenames), cfg.dataset['num_categories'], data_dir))
        
  return filenames, texts, labels

######################################################
# Process a complete data set and save it as a TFRecord.
#   Args:
#    name: string, unique identifier specifying the data set.
#    directory: string, root path to the data set.
#    num_shards: integer number of shards for this data set.
######################################################
def _process_dataset(name, directory, num_shards):

  filenames, texts, labels = _find_image_files(directory)
  _process_image_files(name, filenames, texts, labels, num_shards)


######################################################
######################################################
################         MAIN          ###############
######################################################
######################################################
def main(unused_argv):

  assert not FLAGS.train_shards % FLAGS.num_threads, (
      'Please make the FLAGS.num_threads commensurate with FLAGS.train_shards')
  assert not FLAGS.validation_shards % FLAGS.num_threads, (
      'Please make the FLAGS.num_threads commensurate with '
      'FLAGS.validation_shards')
      
  print('Saving results to %s' % FLAGS.output_directory)

  # Run it!
  _process_dataset('validation', FLAGS.validation_directory, FLAGS.validation_shards)
  _process_dataset('train', FLAGS.train_directory, FLAGS.train_shards)


if __name__ == '__main__':
  tf.app.run()