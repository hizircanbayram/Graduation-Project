'''
UnetDataset_v4 veri kumesini kullanarak ptical flow goruntuleri olusturdugumuzda cok fazla gurultuglu goruntu aldigimizi fark ettik. Bunu onlemek icin yol dusunurken aklima soyle bir fikir geldi: Sonucta calisan bir unet modelim var. Unet'i egitirken kullandigim bu model ile egitme esnasinda kullandigim goruntulerin ve onlarin n frame sonraki goruntulerini modelime sokarak sadece elin civarini alayim. Ardindan sadece el ve civarini iceren goruntulerin motion optical flow goruntulerini olusturayim. Bu sayede sadece el civarindaki optical flow goruntuleri elime gececek. Bu da iki sey icin faydali olabilir:
    1) sadece el civarina odaklanmasina yardim edebilir
    2) el ve civari disindaki gurultulerden kurtulurum  
Buradaki script optical flow goruntulerini icerecek kismi veri kumesini olusturmak icin yazilmistir. HSV olarak kaydedilecek olan bu optical flow goruntulerinin 1. kanalinin 255 ile dolu oldugu ve modeli egitirken kullanilmayacagini da bu vesileyle hatirlatmis olalim.
'''

import os

import cv2
import scipy.misc
import numpy as np

from keras.models import load_model
from keras.preprocessing.image import load_img, save_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from unet20 import unet
from predict import getHandArea, get_mask


def getNextNthFrameName(img_path, nth_frame_for_motion):
    splitted_name = img_path.split('-')
    #print(splitted_name)
    frame_no = str(int(splitted_name[1]) - nth_frame_for_motion)     
    new_path = splitted_name[0] + '-' + frame_no + '-' + splitted_name[2]
    return new_path 


def createOpticalFlowDataset(model, dir_path, img_names, motion_vectors_dir, nth_frame_for_motion):
    for i, img_name in enumerate(img_names):

        if img_name.endswith('.png'):
            continue

        sample_name = dir_path + '/' + img_name
        sample_next_name = 'imgs' + str(nth_frame_for_motion) + '/' + img_name

        sample = img_to_array(load_img(sample_name, color_mode='rgb', interpolation='bilinear', target_size=(params['dim'][0], params['dim'][0], 3)), dtype='uint8')
        sample_next = img_to_array(load_img(getNextNthFrameName(sample_next_name, nth_frame_for_motion), color_mode='rgb', interpolation='bilinear', target_size=(params['dim'][0], params['dim'][0], 3)), dtype='uint8')

        sample_gray = img_to_array(load_img(sample_name, color_mode='grayscale', interpolation='bilinear', target_size=(params['dim'][0], params['dim'][0], 3)), dtype='uint8')
        sample_next_gray = img_to_array(load_img(getNextNthFrameName(sample_next_name, nth_frame_for_motion), color_mode='grayscale', interpolation='bilinear', target_size=(params['dim'][0], params['dim'][0], 3)), dtype='uint8')    

        sample_mask = get_mask(model, sample, 0)
        sample_next_mask = get_mask(model, sample_next, 0)

        img1 = getHandArea(sample, sample_mask)
        img2 = getHandArea(sample_next, sample_next_mask)

        gray_img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        gray_img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

        if nth_frame_for_motion > 0:
            flow = cv2.calcOpticalFlowFarneback(gray_img1,gray_img2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        else:
            flow = cv2.calcOpticalFlowFarneback(gray_img2,gray_img1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        ang = ang*180/np.pi/2 # need to be normalized, just like normal rgb hand images. it will be normalized below.
        mag = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX) # need to be normalized, just like normal rgb hand images. it will be normalized below.
        
        hsv = np.zeros_like(sample)
        hsv[...,1] = 255
        hsv[...,0] = ang
        hsv[...,2] = mag 
        rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        cv2.imwrite(motion_vectors_dir + '/' + img_name, rgb)
        if (i != 0) and (i % 100 == 0):
            print(i // 2, '/', len(img_names) // 2, 'saved')

        ''' UNCOMMENT IF YOU WANT TO DISPLAY AND COMMENT save_img #####
        rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        cv2.imshow('motion time 8)', rgb)
        cv2.waitKey(0)
        ##### UNCOMMENT IF YOU WANT TO DISPLAY AND COMMENT save_img '''





train_path = 'UnetDataset_v4/train'
test_path = 'UnetDataset_v4/test'
validation_path = 'UnetDataset_v4/validation'

nth_frame_for_motion = 2 # 0 veremezsin(bu bir optical flow veri kumesi olusturmak icin tasarlanan script oldugundan dolayi
model_path = 'check_points/unet20_52_256/Unet20_07-0.96.hdf5'
params = {'dim': (256,256,3),'batch_size': 8, 'shuffle': True} 
dir_path = validation_path

train_img_names = os.listdir(train_path)
test_img_names = os.listdir(test_path)
validation_img_names = os.listdir(validation_path)

model = load_model(model_path)
motion_vectors_dir = 'motion_vectors_' + str(nth_frame_for_motion) # destination dir, where the motion vector dataset will be stored

createOpticalFlowDataset(model, dir_path, validation_img_names, motion_vectors_dir, nth_frame_for_motion) # 2. parametre olan dir_path ve 3. parametre olan goruntu isimlerinin eslesmesine dikkat et 
    


