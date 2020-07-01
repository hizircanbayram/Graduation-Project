import numpy as np
import keras
import cv2
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, train_dir, data_type, list_IDs, labels, nth_frame_for_motion=0, optical_flow_dir=None, dim=(32,32,32), batch_size=32, shuffle=True):
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
            dir_name = self.train_dir + self._data_type
            # Image
            x_inp = np.zeros(self.dim)
            sample = img_to_array(load_img(dir_name + ID, color_mode='rgb', interpolation='bilinear', target_size=(self.dim[0], self.dim[1], 3)), dtype='uint8')
 
            x_inp[:,:,0:3] = sample / 255.  
            if self.nth_frame_for_motion > 0: # optical flow
                optical_flow_img = cv2.imread('motion_vectors_' + str(self.nth_frame_for_motion) + '/' + ID)
                optical_flow_img_hsv = cv2.cvtColor(optical_flow_img, cv2.COLOR_BGR2HSV)      
                x_inp[:,:,3] = optical_flow_img_hsv[:,:,0] / 180.
                x_inp[:,:,4] = optical_flow_img_hsv[:,:,2] / 255.
            X.append(x_inp)
   
            '''
            # test for normalized version of motion channels' distribution
            plt.hist(x_inp[:,:,4])
            plt.show()
            cv2.waitKey(0)
            # test for normalized version of motion channels' distribution
            '''
            
            # test for motion images
            #cv2.imshow('motion time 8)', optical_flow_img)
            #cv2.waitKey(0)
            # test for motion images
            
            # optical flow
            #print('X.shape: ', sample.shape)
            if sample is None:
                print('rgb not read: ', dir_name + ID)
            # Mask
            #sample = cv2.imread(dir_name + self.labels[ID], cv2.IMREAD_GRAYSCALE) 
            sample =   self.labels[ID]
            y.append(sample)
            #axarr[0][0].imshow(X[i].astype('uint8'))
            #axarr[0][1].imshow(y[i].astype('uint8'))
            #f.show()
            #plt.waitforbuttonpress()
            
        X = np.array(X)
        y = np.array(y)
        y = y[:, :, :, np.newaxis]
        #y2 = y2[:, :, :, np.newaxis]
        #print('y.shape: ', y.shape)
        #print('y2.shape: ', y2.shape)
        return X, y

