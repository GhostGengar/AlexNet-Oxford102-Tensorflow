#############################################
# Tien Dinh                 		    #
# Friendswood, TX 2/24/2018 	            #
# For full documentation, 		    #
# please refer to the Jupyter Notebook file #
#############################################

import os
from datetime import datetime

import numpy as np
import cv2
from scipy.io import loadmat

import tensorflow as tf
from alexnet import AlexNet
from datapreprocess import ImageProcessor

imagenet_mean = np.array([104., 117., 124.], dtype=np.float32)

os.mkdir('./summary')
os.mkdir('./models')

def main():

	set_ids = loadmat('setid.mat')
	test_ids = set_ids['trnid'].tolist()[0]
	train_ids = set_ids['tstid'].tolist()[0]
	raw_train_ids = indexes_processing(train_ids)
	raw_test_ids = indexes_processing(test_ids)

	image_labels = (loadmat('imagelabels.mat')['labels'] - 1).tolist()[0]

	image_processor = ImageProcessor()
	image_processor.set_up_images()

	x = tf.placeholder(tf.float32, [None, 227, 227, 3])
	y_true = tf.placeholder(tf.float32, [None, 102])
	keep_prob = tf.placeholder(tf.float32)

	global_step = tf.Variable(0, trainable=False)
	base_lr = 0.001
	base_lr = tf.train.exponential_decay(base_lr, global_step, 20000, 0.5, staircase=True)
	num_epochs = 50000
	drop_rate = 0.5
	train_layers = ['fc8']

	model = AlexNet(x, keep_prob, 102, train_layers)

	with tf.name_scope('network_output'):
    	y_pred = model.y_pred

    all_vars = tf.trainable_variables()
	conv_vars = [all_vars[0], all_vars[2], all_vars[4], all_vars[6], all_vars[8], all_vars[10], all_vars[12]]
	bias_vars = [all_vars[1], all_vars[3], all_vars[5], all_vars[7], all_vars[9], all_vars[11], all_vars[13]]
	last_weights = [all_vars[14]]
	last_bias = [all_vars[15]]

	with tf.name_scope('cross_entropy'):
    	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=y_pred))
    tf.summary.scalar('cross_entropy', cross_entropy)

    with tf.name_scope('train'):
    	gradients = tf.gradients(cross_entropy, conv_vars + bias_vars + last_weights + last_bias)
    	conv_vars_gradients = gradients[:len(conv_vars)]
    	bias_vars_gradients = gradients[len(conv_vars):len(conv_vars) + len(bias_vars)]
    	last_weights_gradients = gradients[len(conv_vars) + len(bias_vars):len(conv_vars) + len(bias_vars) + len(last_weights)]
    	last_bias_gradients = gradients[len(conv_vars) + len(bias_vars) + len(last_weights):len(conv_vars) + len(bias_vars) + len(last_weights) + len(last_bias)]
    
	trained_weights_optimizer = tf.train.GradientDescentOptimizer(base_lr)
	trained_biases_optimizer = tf.train.GradientDescentOptimizer(2*base_lr)
	weights_optimizer = tf.train.GradientDescentOptimizer(10*base_lr)
	biases_optimizer = tf.train.GradientDescentOptimizer(20*base_lr)

	train_op1 = trained_weights_optimizer.apply_gradients(zip(conv_vars_gradients, conv_vars))
	train_op2 = trained_biases_optimizer.apply_gradients(zip(bias_vars_gradients, bias_vars))
	train_op3 = weights_optimizer.apply_gradients(zip(last_weights_gradients, last_weights))
	train_op4 = biases_optimizer.apply_gradients(zip(last_bias_gradients, last_bias))

	train = tf.group(train_op1, train_op2, train_op3, train_op4)

	with tf.name_scope('accuracy'):
	    matches = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
	    acc = tf.reduce_mean(tf.cast(matches, tf.float32))
    
	tf.summary.scalar('accuracy', acc)

	merged_summary = tf.summary.merge_all()
	writer = tf.summary.FileWriter('./summary')
	init = tf.global_variables_initializer()
	saver = tf.train.Saver(max_to_keep=3)

	with tf.Session() as sess:
    	sess.run(init)
    	writer.add_graph(sess.graph)
    	model.load_weights(sess)
    
    	print('Training process started at {}'.format(datetime.now()))

    	for i in range(num_epochs):
        	batches = image_processor.next_batch(128)
        	sess.run(train, feed_dict={x:batches[0], y_true:batches[1], keep_prob:0.5})
        	global_step += 1
        	if (i%500==0):
            	print('On Step {}'.format(i))
            	print('Current base learning rate: {0:.5f}'.format(sess.run(base_lr)))
            	print('At: {}'.format(datetime.now()))
            
            	accuracy = sess.run(acc, feed_dict={x:image_processor.testing_images, y_true:image_processor.testing_labels, keep_prob:1.0})
            	print('Accuracy: {0:.2f}%'.format(accuracy * 100))
            
            	print('Saving model...')
            	saver.save(sess, './models/model_iter.ckpt', global_step=i)
            	print('Model saved at step: {}'.format(i))
            	print('\n')
            
    	print('Saving final model...')
    	saver.save(sess, './models/model_final.ckpt')
    	print('Saved')
    	print('Training finished at {}'.format(datetime.now()))

def indexes_processing(int_list):
    returned_list = []
    for index, element in enumerate(int_list):
        returned_list.append(str(element))
    for index, element in enumerate(returned_list):
        if int(element) < 10:
            returned_list[index] = '0000' + element
        elif int(element) < 100:
            returned_list[index] = '000' + element
        elif int(element) < 1000:
            returned_list[index] = '00' + element
        else:
            returned_list[index] = '0' + element
    return returned_list

if __name__ == '__main__':
	main()
