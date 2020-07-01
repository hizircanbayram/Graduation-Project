import os
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.callbacks import ModelCheckpoint

from denseNetGenerator import DataGenerator
from denseNet import build_denseNet




def fill_infos(dataset_dir):
    partition = { 'train': [], 'validation' : [] }
    labels = { }

    train_dir = dataset_dir + '/train' 
    validation_dir = dataset_dir + '/validation'
    
    train_image_names = []
    sample_train_dirs = os.listdir(train_dir)
    for sample_train_dir in sample_train_dirs: # sample_train_dir : directory of one vid
        train_img_names_dir = os.listdir(train_dir + '/' + sample_train_dir) # train_img_names : image names in the directory of that vid
        train_image_names.extend(train_img_names_dir)
    partition['train'].extend(train_image_names)

    validation_image_names = []
    sample_validation_dirs = os.listdir(validation_dir)
    for sample_validation_dir in sample_validation_dirs: # sample_train_dir : directory of one vid
        validation_img_names_dir = os.listdir(validation_dir + '/' + sample_validation_dir) # train_img_names : image names in the directory of that vid
        validation_image_names.extend(validation_img_names_dir)
    partition['validation'].extend(validation_image_names)

    all_image_names = partition['train'].copy()
    all_image_names.extend(partition['validation'])
    for img_name in all_image_names:
        labels[img_name] = getOneHotLabel(img_name)

    return partition, labels



def getOneHotLabel(img_name):
    label = img_name.split('-')[2][:-4]
    if label == 'NoWashing':
        return [1,0,0,0,0,0,0,0,0,0,0,0]
    elif label == 'WetAndApplySoap':
        return [0,1,0,0,0,0,0,0,0,0,0,0] 
    elif label == 'RubPalmToPalm':
        return [0,0,1,0,0,0,0,0,0,0,0,0]
    elif label == 'RubBackOfLeftHand':
        return [0,0,0,1,0,0,0,0,0,0,0,0]
    elif label == 'RubBackOfRightHand':
        return [0,0,0,0,1,0,0,0,0,0,0,0]    
    elif label == 'RubWithInterlacedFingers':
        return [0,0,0,0,0,1,0,0,0,0,0,0]   
    elif label == 'RubWithInterlockedFingers':
        return [0,0,0,0,0,0,1,0,0,0,0,0] 
    elif label == 'RubLeftThumb':
        return [0,0,0,0,0,0,0,1,0,0,0,0] 
    elif label == 'RubRightThumb':
        return [0,0,0,0,0,0,0,0,1,0,0,0] 
    elif label == 'RubRightFingertips':
        return [0,0,0,0,0,0,0,0,0,1,0,0] 
    elif label == 'RubLeftFingerTips':
        return [0,0,0,0,0,0,0,0,0,0,1,0] 
    elif label == 'RinseHands':
        return [0,0,0,0,0,0,0,0,0,0,0,1] 
    else:
        print(label, 'wrong') 





params = {'dim': (256,256,3),
          'batch_size': 8,
          'shuffle': True} 
fpath = "check_points/DenseNetWithoutUnet/" + "DenseNet_{epoch:02d}-{val_accuracy:.2f}.hdf5"
dataset_dir = 'Classifier'

nth_frame = 0
optical_flow_dir = None
partition, labels = fill_infos(dataset_dir)
training_generator   = DataGenerator(dataset_dir, 'train', partition['train'], labels, nth_frame, optical_flow_dir, **params)
validation_generator = DataGenerator(dataset_dir, 'validation', partition['validation'], labels, nth_frame, optical_flow_dir, **params)

check_point = ModelCheckpoint(fpath, monitor='val_accuracy',
                              verbose=2, save_best_only=True, mode='max')

model = build_denseNet(params['dim'], 12)






#model.load_weights('check_points/unet20_30_256_motion_2_normalized/motion_unet22_19-0.96.hdf5')
history = model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    callbacks=[check_point],
                    epochs=50, verbose=1)
                    #use_multiprocessing=True, workers=12)





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
