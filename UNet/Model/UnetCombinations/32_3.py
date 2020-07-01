import os
from unet import unet
from keras.models import Sequential
from data_generator import DataGenerator
from keras.callbacks import ModelCheckpoint
from measure_jaccard import iou_loss_core
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
params = {'dim': (256,256,3),
          'batch_size': 4,
          'shuffle': True}

training_dir = 'UnetDataset_v3' #
model_name = "unet"

fpath = "check_points/"  + model_name +  "/" + training_dir + "/" + model_name + "_{epoch:02d}" + ".hdf5"
partition, labels = fill_infos(training_dir + '/train', training_dir + '/validation')
training_generator   = DataGenerator(training_dir, 'train', partition['train'], labels, **params)
validation_generator = DataGenerator(training_dir, 'validation', partition['validation'], labels, **params)

#fpath = "check_points/small_unet_v2/UnetDataset_v4/" + training_dir + "_small_UNet_v2_{epoch:02d}-{val_accuracy:.2f}_100.hdf5"


check_point = ModelCheckpoint(fpath, monitor='val_iou_loss_core',
                              verbose=2, save_best_only=True, mode='max')
model = unet(input_size=params['dim'])  
from keras.models import load_model
dependencies = {
    'iou_loss_core': iou_loss_core
}
#model = load_model('check_points/unet20/UnetDataset_v1/unet20_20.hdf5', custom_objects=dependencies)

history = model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    callbacks=[check_point],
                    epochs=20, verbose=2,
                    use_multiprocessing=True, workers=4)


# summarize history for accuracy
plt.plot(history.history['iou_loss_core'])
plt.plot(history.history['val_iou_loss_core'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("check_points/"  + model_name +  "/" + training_dir + "/" + model_name + "_accuracy.png")
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("check_points/"  + model_name +  "/" + training_dir + "/" + model_name + "_loss.png")


from numeric_test import jaccard_general
jaccard_general(model, training_dir + '/train', 0, params['dim'][0], None)
jaccard_general(model, training_dir + '/test', 0, params['dim'][0], None)
jaccard_general(model, training_dir + '/validation', 0, params['dim'][0], None) 



