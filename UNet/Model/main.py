import os
from small_unet_v2 import unet
from keras.models import Sequential
from data_generator import DataGenerator
from keras.callbacks import ModelCheckpoint


def fill_infos(train_dir, validation_dir):
    partition = { 'train': [], 'validation' : [] }
    labels = { }

    train_image_names = os.listdir(train_dir)
    train_image_names = [img_name for img_name in train_image_names if img_name.endswith('.jpg')] 
    partition['train'].extend(train_image_names)

    validation_image_names = os.listdir(validation_dir)
    validation_image_names = [img_name for img_name in validation_image_names if img_name.endswith('.jpg')] 
    partition['validation'].extend(validation_image_names)

    all_image_names = partition['train'].copy()
    all_image_names.extend(partition['validation'])
    for img_name in all_image_names:
        labels[img_name] = img_name[:-4] + '-colormask.png'

    return partition, labels





# Parameters
params = {'dim': (512,512,3),
          'batch_size': 4,
          'shuffle': True}

training_dir = 'UnetDataset_v4' # v0 DIRECTLY
partition, labels = fill_infos(training_dir + '/train', training_dir + '/validation')
training_generator   = DataGenerator(training_dir, 'train', partition['train'], labels, **params)
validation_generator = DataGenerator(training_dir, 'validation', partition['validation'], labels, **params)

fpath = "check_points/small_unet_v2/UnetDataset_v4/" + training_dir + "_small_UNet_v2_{epoch:02d}-{val_accuracy:.2f}_100.hdf5"
check_point = ModelCheckpoint(fpath, monitor='val_accuracy',
                              verbose=2, save_best_only=True, mode='max')
model = unet(input_size=params['dim'])  

model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    callbacks=[check_point],
                    epochs=100, verbose=1,
                    use_multiprocessing=True, workers=12)


'''
x, y = training_generator.__getitem__(0)
print('x.shape: ', x.shape)
print('y.shape: ', y.shape)

import random
import matplotlib.pyplot as plt
import numpy as np
r = random.randint(0, len(x)-1)

fig = plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.4)
ax = fig.add_subplot(1, 2, 1)
ax.imshow(x[r])
ax = fig.add_subplot(1, 2, 2)
ax.imshow(np.reshape(y[r], (512, 512)), cmap="gray")
plt.waitforbuttonpress()
'''
