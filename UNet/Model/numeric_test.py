import numpy as np
from keras.models import load_model
from measure_jaccard import query_img_names, measure_jaccard, iou_loss_core


def jaccard_general(model, path):
    print(path)
    names = query_img_names(path, 0)
    print('0 jaccard score: ', measure_jaccard(model, names))
    names = query_img_names(path, 1)
    print('1 jaccard score: ', measure_jaccard(model, names))
    #names = query_img_names(path, 2)
    #print('2 jaccard score: ', measure_jaccard(model, names))
    names = query_img_names(path, 3)
    print('3 jaccard score: ', measure_jaccard(model, names))
    names = query_img_names(path, 4)
    print('4 jaccard score: ', measure_jaccard(model, names))
    names = query_img_names(path, 5)
    print('5 jaccard score: ', measure_jaccard(model, names))
    names = query_img_names(path, 6)
    print('6 jaccard score: ', measure_jaccard(model, names))
    names = query_img_names(path, 7)
    print('7 jaccard score: ', measure_jaccard(model, names))
    names = query_img_names(path, 8)
    print('8 jaccard score: ', measure_jaccard(model, names))
    names = query_img_names(path, 9)
    print('9 jaccard score: ', measure_jaccard(model, names))
    names = query_img_names(path, 10)
    print('10 jaccard score: ', measure_jaccard(model, names))


'''
main_dir_path = 'UnetDataset_v1/'
model_path = 'mnew.hdf5'
#model_path = 'check_points/small_unet_v2/' + main_dir_path + 'UnetDataset_v4_small_UNet_v2_13-0.90.hdf5'

train_path = main_dir_path + 'train'
validation_path = main_dir_path + 'validation'
test_path = main_dir_path + 'test'
dependencies = {
    'iou_loss_core': iou_loss_core
}
model = load_model(model_path, custom_objects=None)
jaccard_general(model, train_path)
'''

