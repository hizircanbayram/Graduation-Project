from keras.preprocessing.image import load_img, img_to_array, save_img
import os

def func(dir_name):
    img_names = os.listdir(dir_name)
    normal_names = []
    for img_name in img_names:
        if img_name.endswith('.png'):
            normal_names.append(img_name)

    for m, normal_name in enumerate(normal_names):
        normal_img = img_to_array(load_img(dir_name + normal_name, color_mode='grayscale', interpolation='bilinear')).astype(int)
        for i in range(normal_img.shape[0]):
            for k in range(normal_img.shape[1]):
                if normal_img[i][k] >= 128:
                    normal_img[i][k] = 255
                else:
                    normal_img[i][k] = 0
        save_img(dir_name + normal_name, normal_img)
        if m != 0 and m % 100 == 0:
            print(m, 'is done')
        m += 1
    print(dir_name, 'is done')

func('UnetDataset_v4/train/')
func('UnetDataset_v4/validation/')
func('UnetDataset_v4/test/')

func('UnetDataset_v3/train/')
func('UnetDataset_v3/validation/')
func('UnetDataset_v3/test/')

func('UnetDataset_v2/train/')
func('UnetDataset_v2/validation/')
func('UnetDataset_v2/test/')

func('UnetDataset_v1/train/')
func('UnetDataset_v1/validation/')
func('UnetDataset_v1/test/')
