#############################
# Tien Dinh                 #
# Friendswood, TX 2/24/2018 #
#############################

import tensorflow as tf
import numpy as np
import cv2

from alexnet import AlexNet
from class_labels import class_names

# Create placeholders for the input and hold prob
x = tf.placeholder(tf.float32, shape=[None, 227, 227, 3])
keep_prob = tf.placeholder(tf.float32)

# The mean of ImageNet
mean = np.array([104., 117., 124.], dtype=np.float32)

# Load the image, resize, and normalize
image = np.zeros((1, 227, 227, 3))
img = cv2.imread('./images/flower.jpg')
img = cv2.resize(img, (227, 227)).astype(np.float32)
img -= mean
image[0] = img

# Initialize the AlexNet model
model = AlexNet(x, keep_prob, 102, [])

# Create and run the session
saver = tf.train.Saver()
sess = tf.Session()
saver.restore(sess, './models/model_iter.ckpt-16000')
out = sess.run(model.y_pred_softmax, feed_dict={x:image, keep_prob:1})[0]
preds = (np.argsort(out)[::-1])[0:5]
for p in preds:
	print(class_names[p], out[p])