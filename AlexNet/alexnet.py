import tensorflow as tf
import numpy as np

class AlexNet():
    
    def __init__(self, X, keep_prob, num_classes, skip_layer, weights_path='DEFAULT'):
        self.X = X
        self.KEEP_PROB = keep_prob
        self.NUM_CLASSES = num_classes
        self.SKIP_LAYER = skip_layer
        if weights_path == 'DEFAULT':
            self.WEIGHTS_PATH = 'bvlc_alexnet.npy'
        else:
            self.WEIGHTS_PATH = weights_path
        
        self.initialize()
        
    def initialize(self):
        
        # 1st Layer: Conv (w ReLu) -> Lrn -> Pool
        conv_1 = self.conv_layer(self.X, 11, 11, 96, 4, 4, name='conv1', padding='VALID')
        norm_1 = self.lrn(conv_1, 2, 1e-05, 0.75, name='norm1')
        pool_1 = self.max_pool(norm_1, 3, 3, 2, 2, name='pool1', padding='VALID')
        
        # 2nd Layer: Conv (w ReLu) -> Lrn -> Pool
        conv_2 = self.conv_layer(pool_1, 5, 5, 256, 1, 1, name='conv2', groups=2)
        norm_2 = self.lrn(conv_2, 2, 1e-05, 0.75, name='norm2')
        pool_2 = self.max_pool(norm_2, 3, 3, 2, 2, name='pool2', padding='VALID')
        
        # 3rd Layer: Conv (w ReLu)
        conv_3 = self.conv_layer(pool_2, 3, 3, 384, 1, 1, name='conv3')

        # 4th Layer: Conv (w ReLu)
        conv_4 = self.conv_layer(conv_3, 3, 3, 384, 1, 1, name='conv4', groups=2)

        # 5th Layer: Conv (w ReLu) -> Pool
        conv_5 = self.conv_layer(conv_4, 3, 3, 256, 1, 1, name='conv5', groups=2)
        pool_5 = self.max_pool(conv_5, 3, 3, 2, 2, name='pool5', padding='VALID')

        # 6th Layer: Flatten -> FC (w ReLu) -> Dropout
        pool_6_flat = tf.reshape(pool_5, [-1, 6*6*256])
        full_6 = self.fully_connected(pool_6_flat, 6*6*256, 4096, name='fc6')
        full_6_dropout = self.drop_out(full_6, self.KEEP_PROB)
        
        # 7th Layer: FC (w ReLu) -> Dropout
        full_7 = self.fully_connected(full_6_dropout, 4096, 4096, name='fc7')
        full_7_dropout = self.drop_out(full_7, self.KEEP_PROB)
        
        # 8th Layer: FC and return unscaled activations
        self.y_pred = self.fully_connected(full_7_dropout, 4096, self.NUM_CLASSES, relu=False, name='fc8')

        # Softmax Layer:
        self.y_pred_softmax = tf.nn.softmax(self.y_pred)
        
    def load_weights(self, session):
        
        # Load the weights into memory
        weights_dict = np.load(self.WEIGHTS_PATH, encoding='bytes').item()
        
        # Loop over all layer names stored in the weights dict
        for op_name in weights_dict:
            # Check if layer should be trained from scratch
            if op_name not in self.SKIP_LAYER:
                with tf.variable_scope(op_name, reuse=True):
                    for data in weights_dict[op_name]:
                        if len(data.shape) == 1:
                            var = tf.get_variable('biases')
                            session.run(var.assign(data))
                        else:
                            var = tf.get_variable('weights')
                            session.run(var.assign(data))
                            
    def conv_layer(self, x, filter_height, filter_width, num_filters, stride_y, stride_x, name, padding='SAME', groups=1):
        num_channels = int(x.get_shape()[-1])
        convolve = lambda i, k: tf.nn.conv2d(i, k, strides=[1,stride_y,stride_x,1], padding=padding)
        with tf.variable_scope(name) as scope:
            weights = tf.get_variable('weights', shape=[filter_height,
                                                        filter_width,
                                                        num_channels/groups,
                                                        num_filters])
            biases = tf.get_variable('biases', shape=[num_filters])
        if groups == 1:
            conv = convolve(x, weights)
        else:
            input_groups = tf.split(axis=3, num_or_size_splits=groups, value=x)
            weight_groups = tf.split(axis=3, num_or_size_splits=groups, value=weights)
            output_groups = [convolve(i, k) for i, k in zip(input_groups, weight_groups)]
            conv = tf.concat(axis=3, values=output_groups)
        bias = tf.reshape(tf.nn.bias_add(conv, biases), tf.shape(conv))
        return tf.nn.relu(bias, name=scope.name)

    def max_pool(self, x, filter_height, filter_width, stride_y, stride_x, name, padding='SAME'):
        return tf.nn.max_pool(x, ksize=[1,filter_height,filter_width,1], 
                              strides=[1,stride_y,stride_x,1], padding=padding,
                              name=name)

    def lrn(self, x, radius, alpha, beta, name, bias=1.0):
        return tf.nn.local_response_normalization(x, depth_radius=radius, 
                                                  alpha=alpha, beta=beta, 
                                                  bias=bias, name=name)

    def fully_connected(self, input_layer, num_in, num_out, name, relu=True):
        with tf.variable_scope(name) as scope:
            weights = tf.get_variable('weights', shape=[num_in, num_out], trainable=True)
            biases = tf.get_variable('biases', shape=[num_out], trainable=True)
            activation = tf.nn.xw_plus_b(input_layer, weights, biases, name=scope.name)
        if relu:
            return tf.nn.relu(activation)
        else:
            return activation
    
    def drop_out(self, x, keep_prob):
        return tf.nn.dropout(x, keep_prob=keep_prob)