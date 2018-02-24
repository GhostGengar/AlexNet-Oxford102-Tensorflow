#############################################
# Tien Dinh                                 #
# Friendswood, TX 2/24/2018                 #
# For full documentation,                   #
# please refer to the Jupyter Notebook file #
#############################################

import numpy as np
import cv2

class ImageProcessor():
    
    def __init__(self, num_classes=102):           
        self.i = 0
        self.num_classes = num_classes
        
        self.training_images = np.zeros((6149, 227, 227, 3))
        self.training_labels = None
        
        self.testing_images = np.zeros((1020, 227, 227, 3))
        self.testing_labels = None
    
    def one_hot_encode(self, labels):
        '''
        One hot encode the output labels to be numpy arrays of 0s and 1s
        '''
        out = np.zeros((len(labels), self.num_classes))
        for index, element in enumerate(labels):
            out[index, element] = 1
        return out
    
    def set_up_images(self):
        print('Processing Training Images...')
        i = 0
        for element in raw_train_ids:
            img = cv2.imread('/input/image_{}.jpg'.format(element))
            img = cv2.resize(img, (227, 227)).astype(np.float32)
            img -= imagenet_mean
            self.training_images[i] = img
            i += 1
        print('Done!')
        
        i = 0
        print('Processing Testing Images...')
        for element in raw_test_ids:
            img = cv2.imread('/input/image_{}.jpg'.format(element))
            img = cv2.resize(img, (227, 227)).astype(np.float32)
            img -= imagenet_mean
            self.testing_images[i] = img
            i += 1
        print('Done!')
        
        print('Processing Training and Testing Labels...')
        encoded_labels = self.one_hot_encode(image_labels)
        for train_id in train_ids:
            train_labels.append(encoded_labels[train_id - 1])
        for test_id in test_ids:
            test_labels.append(encoded_labels[test_id - 1])
        self.training_labels = train_labels
        self.testing_labels = test_labels
        print('Done!')
        
    def next_batch(self, batch_size):
        x = self.training_images[self.i:self.i + batch_size]
        y = self.training_labels[self.i:self.i + batch_size]
        self.i = (self.i + batch_size) % len(self.training_images)
        return x, y