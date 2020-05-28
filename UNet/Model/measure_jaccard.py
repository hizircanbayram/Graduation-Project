import os
import numpy as np
from PIL import Image
from keras.preprocessing.image import load_img, img_to_array
from predict import get_mask
from keras import backend as K



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



def measure_jaccard(model, img_names):
    from sklearn.metrics import jaccard_score
    preds = []
    grounds = []
    for i, img_name in enumerate(img_names):
        #ground_mask = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
        ground_mask = img_to_array(load_img(img_name, color_mode='grayscale', interpolation='bilinear'), dtype='uint8')
        ground_mask = ground_mask.flatten()
        #img = cv2.imread(img_name[:len(img_name) - 14] + '.jpg', 1)
        img = img_to_array(load_img(img_name[:len(img_name) - 14] + '.jpg', color_mode='rgb', interpolation='bilinear'), dtype='uint8')
        segmented_pred = get_mask(model, img)
        segmented_pred = segmented_pred.flatten()
        
        preds = np.concatenate((np.array(preds), segmented_pred))
        grounds = np.concatenate((np.array(grounds), ground_mask))
    preds = preds / 255 # 0 ve 255 sayilari, 0 ve 1 sayilarina indirgensinki jaccard_score fonksiyonu binary modda calisabilsin
    grounds = np.around(grounds / 255)
    return jaccard_score(y_pred=preds, y_true=grounds)



