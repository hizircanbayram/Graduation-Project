from keras.preprocessing.image import load_img, img_to_array, save_img
import os
import random

file_names = os.listdir('train')
i = 0


for file_name in file_names:
    if file_name.endswith('.jpg'):
        if i < 2000:
            normal_image = img_to_array(load_img('train/' + file_name, color_mode='rgb', target_size=(512,512), interpolation='bilinear'))
            segmented_image = img_to_array(load_img('train/' + file_name[:-4] + '-colormask.png', color_mode='grayscale', target_size=(512,512), interpolation='bilinear'))
            save_img('UnetDataset_v3/train' + '/' + file_name, normal_image)
            save_img('UnetDataset_v3/train' + '/' + file_name[:-4] + '-colormask.png', segmented_image)
            i += 1
        if i >= 2000 and i < 4000:
            normal_image = img_to_array(load_img('train/' + file_name, color_mode='rgb', target_size=(512,512), interpolation='bilinear'))
            segmented_image = img_to_array(load_img('train/' + file_name[:-4] + '-colormask.png', color_mode='grayscale', target_size=(512,512), interpolation='bilinear'))
            save_img('UnetDataset_v3/train' + '/' + file_name, normal_image)
            save_img('UnetDataset_v3/train' + '/' + file_name[:-4] + '-colormask.png', segmented_image)

            normal_image = img_to_array(load_img('train/' + file_name, color_mode='rgb', target_size=(512,512), interpolation='bilinear'))
            segmented_image = img_to_array(load_img('train/' + file_name[:-4] + '-colormask.png', color_mode='grayscale', target_size=(512,512), interpolation='bilinear'))
            save_img('UnetDataset_v2/train' + '/' + file_name, normal_image)
            save_img('UnetDataset_v2/train' + '/' + file_name[:-4] + '-colormask.png', segmented_image)
            i += 1
        if i >= 4000 and i < 6000:
            normal_image = img_to_array(load_img('train/' + file_name, color_mode='rgb', target_size=(512,512), interpolation='bilinear'))
            segmented_image = img_to_array(load_img('train/' + file_name[:-4] + '-colormask.png', color_mode='grayscale', target_size=(512,512), interpolation='bilinear'))
            save_img('UnetDataset_v3/train' + '/' + file_name, normal_image)
            save_img('UnetDataset_v3/train' + '/' + file_name[:-4] + '-colormask.png', segmented_image)

            normal_image = img_to_array(load_img('train/' + file_name, color_mode='rgb', target_size=(512,512), interpolation='bilinear'))
            segmented_image = img_to_array(load_img('train/' + file_name[:-4] + '-colormask.png', color_mode='grayscale', target_size=(512,512), interpolation='bilinear'))
            save_img('UnetDataset_v2/train' + '/' + file_name, normal_image)
            save_img('UnetDataset_v2/train' + '/' + file_name[:-4] + '-colormask.png', segmented_image)

            normal_image = img_to_array(load_img('train/' + file_name, color_mode='rgb', target_size=(512,512), interpolation='bilinear'))
            segmented_image = img_to_array(load_img('train/' + file_name[:-4] + '-colormask.png', color_mode='grayscale', target_size=(512,512), interpolation='bilinear'))
            save_img('UnetDataset_v1/train' + '/' + file_name, normal_image)
            save_img('UnetDataset_v1/train' + '/' + file_name[:-4] + '-colormask.png', segmented_image)
            i += 1
        if i == 6000:
            break

