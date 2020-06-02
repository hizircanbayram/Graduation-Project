import os
from unet20 import unet
from keras.models import Sequential
from motion_data_generator import MotionDataGenerator
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt


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
params = {'dim': (256,256,5),
          'batch_size': 8,
          'shuffle': True} 
nth_frame = 2
optical_flow_dir = 'optical_flow_imgs_' + str(nth_frame) + '/'
fpath = "check_points/unet20_50_256_motion_" + str(nth_frame) + '_normalized/' + "without_normalized_motion_unet20_{epoch:02d}-{val_accuracy:.2f}.hdf5"


training_dir = 'UnetDataset_v4'
partition, labels = fill_infos(training_dir + '/train', training_dir + '/validation')
training_generator   = MotionDataGenerator(training_dir, 'train', partition['train'], labels, nth_frame, optical_flow_dir, **params)
validation_generator = MotionDataGenerator(training_dir, 'validation', partition['validation'], labels, nth_frame, optical_flow_dir, **params)

check_point = ModelCheckpoint(fpath, monitor='val_accuracy',
                              verbose=2, save_best_only=True, mode='max')
model = unet(input_size=params['dim'])  

history = model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    callbacks=[check_point],
                    epochs=50, verbose=1,
                    use_multiprocessing=True, workers=4)

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
