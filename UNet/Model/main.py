import os
from unet import unet
from keras.models import Sequential
from motion_data_generator import DataGenerator
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
params = {'dim': (256,256,5),
          'batch_size': 8,
          'shuffle': True}

training_dir = 'UnetDataset_v4' # v0 DIRECTLY
partition, labels = fill_infos(training_dir + '/train', training_dir + '/validation')
training_generator   = DataGenerator(training_dir, 'train', partition['train'], labels, **params)
validation_generator = DataGenerator(training_dir, 'validation', partition['validation'], labels, **params)

fpath = "check_points/small_unet_v2/UnetDataset_v4/" + training_dir + "motion_unet20_{epoch:02d}-{val_accuracy:.2f}.hdf5"

check_point = ModelCheckpoint(fpath, monitor='val_accuracy',
                              verbose=2, save_best_only=True, mode='max')
model = unet(input_size=params['dim'])  

model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    callbacks=[check_point],
                    epochs=50, verbose=1,
                    use_multiprocessing=True, workers=4)


