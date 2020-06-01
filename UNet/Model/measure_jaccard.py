import os
import numpy as np
from PIL import Image
from keras.preprocessing.image import load_img, img_to_array
from predict import get_mask
from keras import backend as K
import cv2


def iou_loss_core(y_true, y_pred, smooth=1):
    #https://www.kaggle.com/c/data-science-bowl-2018/discussion/51553
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    union = K.sum(y_true,-1) + K.sum(y_pred,-1) - intersection
    iou = (intersection + smooth) / ( union + smooth)
    return iou



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



def measure_jaccard(model, img_names, nth_frame, img_width_height, optical_flow_dir):
    from sklearn.metrics import jaccard_score
    preds = []
    grounds = []
    for i, img_name in enumerate(img_names):
        ground_mask = img_to_array(load_img(img_name, color_mode='grayscale', interpolation='bilinear', target_size=(img_width_height,img_width_height,3)), dtype='uint8')
        ground_mask = ground_mask.flatten()
        rgb_img_name = img_name[:len(img_name) - 14] + '.jpg'
        sample = img_to_array(load_img(rgb_img_name, color_mode='rgb', interpolation='bilinear', target_size=(img_width_height,img_width_height,3)), dtype='uint8')
        x_inp = None
        ########
        if nth_frame > 0:
            x_inp = np.zeros((img_width_height, img_width_height, 5))
            sample_gray = img_to_array(load_img(rgb_img_name, color_mode='grayscale', interpolation='bilinear', target_size=(img_width_height,img_width_height,3)), dtype='uint8')
            only_img_name = rgb_img_name.split('/')[2]
            splitted_name = only_img_name.split('-')
            next_frame_name = optical_flow_dir + splitted_name[0] + '-' + str(int(splitted_name[1]) + nth_frame) + '-' + splitted_name[2]
            sample_gray_next = img_to_array(load_img(next_frame_name, color_mode='grayscale', interpolation='bilinear', target_size=(img_width_height,img_width_height,3)), dtype='uint8')
            flow = cv2.calcOpticalFlowFarneback(sample_gray,sample_gray_next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
            ang = ang*180/np.pi/2 # need to be normalized, just like normal rgb hand images. it will be normalized below.
            mag = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX) # need to be normalized, just like normal rgb hand images. it will be normalized below.
            # UnetDataset_v4/test/HIJEL_3004 (12)-506-0.jpg
            x_inp[:,:,0:3] = sample 
            x_inp[:,:,3] = ang
            x_inp[:,:,4] = mag 
        else:
            x_inp = sample
        ########     
        segmented_pred = get_mask(model, x_inp, nth_frame)
        segmented_pred = segmented_pred.flatten()
        
        preds = np.concatenate((np.array(preds), segmented_pred))
        grounds = np.concatenate((np.array(grounds), ground_mask))
    preds = preds / 255 # 0 ve 255 sayilari, 0 ve 1 sayilarina indirgensinki jaccard_score fonksiyonu binary modda calisabilsin
    grounds = np.around(grounds / 255)
    return jaccard_score(y_pred=preds, y_true=grounds)



