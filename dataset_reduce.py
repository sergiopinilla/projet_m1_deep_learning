# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A very simple MNIST classifier.

See extensive documentation at
https://www.tensorflow.org/get_started/mnist/beginners
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
# ##############################################
import random
# ###############################################
FLAGS = None


def main(_):
  # ###########################################################
  # This function allows to choose randomly a batch from a smaller dataset
  # Input: fx_cut and fy_cut are numpy arrays of shape (si_dset,784) and 
  # (si_dset,10) respectively. fba_si is the batch size. si_dset and fba_si
  # are defined in the parameters section.
  def choose_random_mnist(fx_cut,fy_cut,fba_si):
    si_dset = len(fx_cut) # I recover si_dset (size of the dataset).
    ve_rsi_dset = range(si_dset) # I create a vector [0,1,2,...,si_dset-1].
    ran_li = random.sample(ve_rsi_dset,fba_si) # I create another vector with
                                               # fba_si (batch size) elements
                                               # randomly chosen from ve_rsi_dset.
    fx_b = fx_cut[ran_li] # I select the elements indexed by ran_li out of fx_cut.
    fy_b = fy_cut[ran_li] # Idem.
    return (fx_b, fy_b) # Output.
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
  # ################################################################
  # Number of times we will run the program in order to make a statistics.
  n_times = 4
  # This is the vector of the sizes we will consider for the dataset.
  vec_sdset = [100,500,1000,5000,10000,25000,50000]
  # This list will contain all of the precisions.
  big_pre = []
  # I will write our results on this file.
  wfile=open('prec_vs_dset_size.dat','w')
  for k in range(n_times): # loop over the number of times
    prec = [] # we will store temporarily the precisions on this vector
    for sdse in vec_sdset: # loop over the size of the dataset
      # #########################################################
      # #########################################################
      # PARAMETERS
      si_dset = sdse # Size of the dataset. It should be smaller than 55000.
      ba_si = 100 # Batch size. It should be smaller than si_dset.
      num_loops = 1000 # Number of loops for the training part.
      # #########################################################

      # Create the model
      x = tf.placeholder(tf.float32, [None, 784])
      W = tf.Variable(tf.zeros([784, 10]))
      b = tf.Variable(tf.zeros([10]))
      y = tf.matmul(x, W) + b

      # Define loss and optimizer
      y_ = tf.placeholder(tf.float32, [None, 10])
      # #############################################################
      # I create a new dataset using the methods of tf.
      # I create a batch out of the 55000 images
      # and store it (this is my new dataset).
      x_cut, y_cut = mnist.train.next_batch(si_dset)
      # #############################################################
      # The raw formulation of cross-entropy,
      #
      #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
      #                                 reduction_indices=[1]))
      #
      # can be numerically unstable.
      #
      # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
      # outputs of 'y', and then average across the batch.
      cross_entropy = tf.reduce_mean(
          tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
      train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

      sess = tf.InteractiveSession()
      tf.global_variables_initializer().run()
      # Train
      for _ in range(num_loops):
        batch_xs, batch_ys = choose_random_mnist(x_cut,y_cut,ba_si)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

      # Test trained model
      correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
      # I store the precisions in the list prec.
      prec.append(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                      y_: mnist.test.labels}))
    # I store the precisions for the whole set of dataset sizes in big_pre and
    # then we start again.
    big_pre.append(prec)
  # I write properly our results on the file.
  for m in range(len(vec_sdset)):
    wfile.write(str(vec_sdset[m])+' ')
    for n in range(n_times):
      wfile.write(str(big_pre[n][m])+' ')
    wfile.write('\n')
  # I close the file.
  wfile.close()
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
