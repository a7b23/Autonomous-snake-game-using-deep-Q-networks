import tensorflow as tf
import numpy as np
import heapq
import random

batchSize = 100
FUTURE_REWARD_DISCOUNT = 0.99
class create_network():
	def __init__(self,numberActions,learningRate):
		self.convolution_weights_1 = tf.Variable(tf.truncated_normal([8, 8, 4, 16], stddev=0.01))
		self.convolution_bias_1 = tf.Variable(tf.constant(0.01, shape=[16]))

		self.convolution_weights_2 = tf.Variable(tf.truncated_normal([4, 4, 16, 32], stddev=0.01))
		self.convolution_bias_2 = tf.Variable(tf.constant(0.01, shape=[32]))

		self.convolution_weights_3 = tf.Variable(tf.truncated_normal([3, 3, 32, 32], stddev=0.01))
		self.convolution_bias_3 = tf.Variable(tf.constant(0.01, shape=[32]))

		self.feed_forward_weights_1 = tf.Variable(tf.truncated_normal([800, 256], stddev=0.01))
		self.feed_forward_bias_1 = tf.Variable(tf.constant(0.01, shape=[256]))

		self.feed_forward_weights_2 = tf.Variable(tf.truncated_normal([256, 4], stddev=0.01))
		self.feed_forward_bias_2 = tf.Variable(tf.constant(0.01, shape=[4]))

		self.input_layer = tf.placeholder("float", [None, 80, 80,4])
		                                    

		self.hidden_convolutional_layer_1 = tf.nn.relu(
		    tf.nn.conv2d(self.input_layer, self.convolution_weights_1, strides=[1, 4, 4, 1], padding="SAME") + self.convolution_bias_1)

		#self.hidden_max_pooling_layer_1 = tf.nn.max_pool(self.hidden_convolutional_layer_1, ksize=[1, 2, 2, 1],
		#                                            strides=[1, 2, 2, 1], padding="SAME")

		self.hidden_convolutional_layer_2 = tf.nn.relu(
		    tf.nn.conv2d(self.hidden_convolutional_layer_1, self.convolution_weights_2, strides=[1, 2, 2, 1],
		                 padding="SAME") + self.convolution_bias_2)

		#self.hidden_max_pooling_layer_2 = tf.nn.max_pool(self.hidden_convolutional_layer_2, ksize=[1, 2, 2, 1],
		#                                            strides=[1, 2, 2, 1], padding="SAME")

		self.hidden_convolutional_layer_3 = tf.nn.relu(
		    tf.nn.conv2d(self.hidden_convolutional_layer_2, self.convolution_weights_3,
		                 strides=[1, 2, 2, 1], padding="SAME") + self.convolution_bias_3)

		#self.hidden_max_pooling_layer_3 = tf.nn.max_pool(self.hidden_convolutional_layer_3, ksize=[1, 2, 2, 1],
		#                                            strides=[1, 2, 2, 1], padding="SAME")

		
		self.hidden_convolutional_layer_3_flat = tf.reshape(self.hidden_convolutional_layer_3, [-1, 800])

		self.final_hidden_activations = tf.nn.relu(
		    tf.matmul(self.hidden_convolutional_layer_3_flat, self.feed_forward_weights_1) + self.feed_forward_bias_1)

		self.output_layer = tf.matmul(self.final_hidden_activations, self.feed_forward_weights_2) + self.feed_forward_bias_2
		self.predict_action = tf.argmax(self.output_layer,1)
		
		self.feedAction = tf.placeholder("float", [None, numberActions])
		self.target = tf.placeholder("float", [None])
		self.readout_action = tf.reduce_sum(tf.mul(self.output_layer, self.feedAction), reduction_indices=1)
		self.cost = tf.reduce_mean(tf.square(self.target - self.readout_action))
		self.train_operation = tf.train.AdamOptimizer(learningRate).minimize(self.cost)		



