import numpy as np
import keras
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, train_dir, data_type, list_IDs, labels, batch_size=32, dim=(32,32,32), shuffle=True):
        'Initialization'
        self.train_dir = train_dir
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.data_type = data_type
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
            sample = img_to_array(load_img(dir_name + ID, color_mode='rgb', interpolation='bilinear'), dtype='uint8')
            #print('X.shape: ', sample.shape)
            if sample is None:
                print('rgb not read: ', dir_name + ID)
            sample = sample / 255.0
            X.append(sample)
            # Mask
            #sample = cv2.imread(dir_name + self.labels[ID], cv2.IMREAD_GRAYSCALE) 
            sample =   img_to_array(load_img(dir_name + self.labels[ID], color_mode='grayscale', interpolation='bilinear'), dtype='uint8')[:,:,0]
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

