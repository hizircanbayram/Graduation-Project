import os
from keras.models import load_model
import cv2
import numpy as np

def query_img_names(path=None, movement_no=None):
    img_names = os.listdir(path)
    wished_names = []
    for img_name in img_names:
        if img_name.endswith('.png'):
            if (movement_no is not None):
                splitted = img_name.split('-')
                index = int(splitted[len(splitted) - 2])        
                if index == movement_no:
                    wished_names.append(path + '/' + img_name)
            else:
                wished_names.append(path + '/' + img_name)

    return wished_names


def get_mask(model, img):
    # BGR goruntu ile calisir
    img = img / 255.0
    img = img[np.newaxis, :, :, :]        
    segmented_pred = model.predict(img)
    segmented_pred = segmented_pred[0]
    segmented_pred = segmented_pred > 0.5
    segmented_pred = segmented_pred * 255

    return segmented_pred


def measure_kappa(model, img_names):
    from sklearn.metrics import jaccard_score
    preds = []
    grounds = []
    for img_name in img_names:
        ground_mask = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
        ground_mask = ground_mask.flatten()
        img = cv2.imread(img_name[:len(img_name) - 14] + '.jpg', 1)
        segmented_pred = get_mask(model, img)
        segmented_pred = segmented_pred.flatten()
        
        preds = np.concatenate((np.array(preds), segmented_pred))
        grounds = np.concatenate((np.array(grounds), ground_mask))

    preds = preds / 255
    grounds = np.around(grounds / 255)

    return jaccard_score(y_pred=preds, y_true=grounds)



main_dir_path = 'UnetDataset_v4/'
model_path = 'check_points/classic_unet/UnetDataset_v4_UNet_07-0.91.hdf5'

train_path = main_dir_path + 'train'
validation_path = main_dir_path + 'validation'
test_path = main_dir_path + 'test'


model = load_model(model_path)
#print(model.summary())

names = query_img_names(train_path, 0)
print('train 0 jaccard score: ', measure_kappa(model, names))
names = query_img_names(train_path, 1)
print('train 1 jaccard score: ', measure_kappa(model, names))
names = query_img_names(train_path, 2)
print('train 2 jaccard score: ', measure_kappa(model, names))
names = query_img_names(train_path, 3)
print('train 3 jaccard score: ', measure_kappa(model, names))
names = query_img_names(train_path, 4)
print('train 4 jaccard score: ', measure_kappa(model, names))
names = query_img_names(train_path, 5)
print('train 5 jaccard score: ', measure_kappa(model, names))
names = query_img_names(train_path, 6)
print('train 6 jaccard score: ', measure_kappa(model, names))
names = query_img_names(train_path, 7)
print('train 7 jaccard score: ', measure_kappa(model, names))
names = query_img_names(train_path, 8)
print('train 8 jaccard score: ', measure_kappa(model, names))
names = query_img_names(train_path, 9)
print('train 9 jaccard score: ', measure_kappa(model, names))
names = query_img_names(train_path, 10)
print('train 10 jaccard score: ', measure_kappa(model, names))


names = query_img_names(validation_path, 0)
print('validation 0 jaccard score: ', measure_kappa(model, names))
names = query_img_names(validation_path, 1)
print('validation 1 jaccard score: ', measure_kappa(model, names))
names = query_img_names(validation_path, 2)
print('validation 2 jaccard score: ', measure_kappa(model, names))
names = query_img_names(validation_path, 3)
print('validation 3 jaccard score: ', measure_kappa(model, names))
names = query_img_names(validation_path, 4)
print('validation 4 jaccard score: ', measure_kappa(model, names))
names = query_img_names(validation_path, 5)
print('validation 5 jaccard score: ', measure_kappa(model, names))
names = query_img_names(validation_path, 6)
print('validation 6 jaccard score: ', measure_kappa(model, names))
names = query_img_names(validation_path, 7)
print('validation 7 jaccard score: ', measure_kappa(model, names))
names = query_img_names(validation_path, 8)
print('validation 8 jaccard score: ', measure_kappa(model, names))
names = query_img_names(validation_path, 9)
print('validation 9 jaccard score: ', measure_kappa(model, names))
names = query_img_names(validation_path, 10)
print('validation 10 jaccard score: ', measure_kappa(model, names))


names = query_img_names(test_path, 0)
print('test 0 jaccard score: ', measure_kappa(model, names))
names = query_img_names(test_path, 1)
print('test 1 jaccard score: ', measure_kappa(model, names))
names = query_img_names(test_path, 2)
print('test 2 jaccard score: ', measure_kappa(model, names))
names = query_img_names(test_path, 3)
print('test 3 jaccard score: ', measure_kappa(model, names))
names = query_img_names(test_path, 4)
print('test 4 jaccard score: ', measure_kappa(model, names))
names = query_img_names(test_path, 5)
print('test 5 jaccard score: ', measure_kappa(model, names))
names = query_img_names(test_path, 6)
print('test 6 jaccard score: ', measure_kappa(model, names))
names = query_img_names(test_path, 7)
print('test 7 jaccard score: ', measure_kappa(model, names))
names = query_img_names(test_path, 8)
print('test 8 jaccard score: ', measure_kappa(model, names))
names = query_img_names(test_path, 9)
print('test 9 jaccard score: ', measure_kappa(model, names))
names = query_img_names(test_path, 10)
print('test 10 jaccard score: ', measure_kappa(model, names))



