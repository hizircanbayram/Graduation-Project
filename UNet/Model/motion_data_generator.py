import numpy as np
import keras
import cv2
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img
import matplotlib.pyplot as plt

class MotionDataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, train_dir, data_type, list_IDs, labels, nth_frame_for_motion=1, optical_flow_dir=None, dim=(32,32,32), batch_size=32, shuffle=True):
        'Initialization'
        self.train_dir = train_dir
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.data_type = data_type
        self.nth_frame_for_motion = nth_frame_for_motion
        self.optical_flow_dir = optical_flow_dir
        self.on_epoch_end()


    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.list_IDs) / float(self.batch_size)))


    def __getitem__(self, index):
        'Generate one batch of data'
        if (index + 1) * self.batch_size > len(self.list_IDs):
            self.batch_size = len(self.list_IDs) - (index * self.batch_size)
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)
        return X, y


    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


    def getNextNthFrameName(self, img_path):
        splitted_name = img_path.split('-')
        frame_no = str(int(splitted_name[1]) + self.nth_frame_for_motion)     
        new_path = splitted_name[0] + '-' + frame_no + '-' + splitted_name[2]
        return new_path   


    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        #X = np.empty((self.batch_size, *self.dim))
        #y = np.empty((self.batch_size, 512,512))
        X = []
        y = []
        from matplotlib import pyplot as plt

        #f, axarr = plt.subplots(1,2,squeeze=False)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            if self.data_type == 'train':
                dir_name = self.train_dir + '/train/'
            elif self.data_type == 'validation':
                dir_name = self.train_dir + '/validation/'
            elif self._data_type == 'test':
                dir_name = self.train_dir + '/test/'
            else:
                print('Wrong typing of data_type!!')
                
            # Image
            #sample = cv2.imread(dir_name + ID, 1)
            x_inp = np.zeros(self.dim)
            sample = img_to_array(load_img(dir_name + ID, color_mode='rgb', interpolation='bilinear', target_size=(self.dim[0], self.dim[1], 3)), dtype='uint8')
            # optical flow
            sample_gray = img_to_array(load_img(dir_name + ID, color_mode='grayscale', interpolation='bilinear', target_size=(self.dim[0], self.dim[1], 3)), dtype='uint8')
            sample_gray_next = img_to_array(load_img(self.getNextNthFrameName(self.optical_flow_dir + ID), color_mode='grayscale', interpolation='bilinear', target_size=(self.dim[0], self.dim[1], 3)), dtype='uint8')
            flow = cv2.calcOpticalFlowFarneback(sample_gray,sample_gray_next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
            ang = ang*180/np.pi/2 # need to be normalized, just like normal rgb hand images. it will be normalized below.
            mag = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX) # need to be normalized, just like normal rgb hand images. it will be normalized below.
            x_inp[:,:,0:3] = sample / 255.
            x_inp[:,:,3] = ang / 180.
            x_inp[:,:,4] = mag / 255.
            X.append(x_inp)
            '''
            # test for normalized version of motion channels' distribution
            plt.hist(x_inp[:,:,4])
            plt.show()
            cv2.waitKey(0)
            # test for normalized version of motion channels' distribution
            '''
            '''
            # test for motion images
            hsv = np.zeros_like(sample)
            hsv[...,1] = 255
            hsv[...,0] = ang / 180.
            hsv[...,2] = mag / 255.
            rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
            cv2.imshow('motion time 8)', rgb)
            cv2.waitKey(0)
            # test for motion images
            '''
            # optical flow
            #print('X.shape: ', sample.shape)
            if sample is None:
                print('rgb not read: ', dir_name + ID)
            # Mask
            #sample = cv2.imread(dir_name + self.labels[ID], cv2.IMREAD_GRAYSCALE) 
            sample =   img_to_array(load_img(dir_name + self.labels[ID], color_mode='grayscale', interpolation='bilinear', target_size=(256,256,3)), dtype='uint8')[:,:,0]
            if sample is None:
                print('gray not read: ', dir_name + self.labels[ID])
            sample = sample / 255.0
            y.append(sample)
            
            #axarr[0][0].imshow(X[i].astype('uint8'))
            #axarr[0][1].imshow(y[i].astype('uint8'))
            #f.show()
            #plt.waitforbuttonpress()
            
        X = np.array(X)
        y = np.array(y)
        y = y[:, :, :, np.newaxis]
        return X, y
