import numpy as np
from keras.models import load_model
from measure_jaccard import query_img_names, measure_jaccard, iou_loss_core


def jaccard_general(model, path, nth_frame, img_width_height, optical_flow_dir):
    print(path)
    names = query_img_names(path, 0)
    print('0 jaccard score: ', measure_jaccard(model, names, nth_frame, img_width_height, optical_flow_dir))
    names = query_img_names(path, 1)
    print('1 jaccard score: ', measure_jaccard(model, names, nth_frame, img_width_height, optical_flow_dir))
    #names = query_img_names(path, 2)
    #print('2 jaccard score: ', measure_jaccard(model, names, optical_flow_dir))
    names = query_img_names(path, 3)
    print('3 jaccard score: ', measure_jaccard(model, names, nth_frame, img_width_height, optical_flow_dir))
    names = query_img_names(path, 4)
    print('4 jaccard score: ', measure_jaccard(model, names, nth_frame, img_width_height, optical_flow_dir))
    names = query_img_names(path, 5)
    print('5 jaccard score: ', measure_jaccard(model, names, nth_frame, img_width_height, optical_flow_dir))
    names = query_img_names(path, 6)
    print('6 jaccard score: ', measure_jaccard(model, names, nth_frame, img_width_height, optical_flow_dir))
    names = query_img_names(path, 7)
    print('7 jaccard score: ', measure_jaccard(model, names, nth_frame, img_width_height, optical_flow_dir))
    names = query_img_names(path, 8)
    print('8 jaccard score: ', measure_jaccard(model, names, nth_frame, img_width_height, optical_flow_dir))
    names = query_img_names(path, 9)
    print('9 jaccard score: ', measure_jaccard(model, names, nth_frame, img_width_height, optical_flow_dir))
    names = query_img_names(path, 10)
    print('10 jaccard score: ', measure_jaccard(model, names, nth_frame, img_width_height, optical_flow_dir))
    





nth_frame = 2
#model_path = 'check_points/unet20_52_256/Unet20_07-0.96.hdf5'
model_path = 'check_points/unet20_50_256_motion_' + str(nth_frame) + '_normalized/motion_unet20_27-0.96.hdf5'
img_width_height = 256

main_dir_path = 'UnetDataset_v4/'
optical_flow_dir = 'optical_flow_imgs_' + str(nth_frame) + '/'
train_path = main_dir_path + 'train'
validation_path = main_dir_path + 'validation'
test_path = main_dir_path + 'test'
dependencies = {
    'iou_loss_core': iou_loss_core
}
model = load_model(model_path, custom_objects=None)
jaccard_general(model, test_path, nth_frame, img_width_height, optical_flow_dir)


