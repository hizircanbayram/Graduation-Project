# bu dosyadaki olcumler modelin her harekette eli ne kadar iyi ogrendigini anlamak amaciyla yapilmistir. 
# her el hareketi icin sirasiyla %5'ten baslayarak %100'e kadar jaccard scorelari kontrol edilecek. 100 goruntuden a tanesi %10'dan yuksek jaccard degeri alsin, b tanesi %20'den yuksek jaccard degeri alsin... 
# bu sekilde oran artirilarak modelin her harekette her goruntuyu ogrenip ogrenemedigine bakilacak.

import numpy as np
import cv2

from keras.models import load_model

from measure_jaccard import query_img_names, get_mask, iou_loss_core, measure_jaccard
from predict import getHandArea
from unet20 import unet

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

import matplotlib.pyplot as plt


def measure_accuracy_frame_wise(model, hand_no, nth_frame, img_size, optical_flow_dir):
    print('HAND NO', hand_no)
    names = query_img_names(test_path, hand_no)
    sample_no = len(names)
    scores = []

    scales = [.05, .1, .15, .20, .2, .3, .35, .4, .45, .5, .55, .6, .65, .7, .75, .8, .85, .9, .95, 1.]
    for i, name in enumerate(names):
        jac_scr = measure_jaccard(model, [name], nth_frame, img_size, optical_flow_dir)
        scores.append(jac_scr)
        #print(i, '/', sample_no, name, jac_scr)
        #cv2.imshow('heh', sample)
        #cv2.waitKey(0)

    for scale in scales:
        i = 0
        for score in scores:
            if score > scale:  
                i += 1
        #print(i, '/', sample_no, 'passed %', scale)
        print(i)


model_path = 'check_points/unet20_50_256_motion_2_normalized/motion_unet20_27-0.96.hdf5'
#model_path = 'check_points/unet20_52_256/Unet20_07-0.96.hdf5'
dependencies = {
    'iou_loss_core': iou_loss_core
}
nth_frame = 2

test_path = 'UnetDataset_v4/test'
img_size = 256
optical_flow_dir = 'optical_flow_imgs_' + str(nth_frame) + '/'
model = load_model(model_path, custom_objects=dependencies)
measure_accuracy_frame_wise(model, 0, nth_frame, img_size, optical_flow_dir)
measure_accuracy_frame_wise(model, 1, nth_frame, img_size, optical_flow_dir)
measure_accuracy_frame_wise(model, 3, nth_frame, img_size, optical_flow_dir)
measure_accuracy_frame_wise(model, 4, nth_frame, img_size, optical_flow_dir)
measure_accuracy_frame_wise(model, 5, nth_frame, img_size, optical_flow_dir)
measure_accuracy_frame_wise(model, 6, nth_frame, img_size, optical_flow_dir)
measure_accuracy_frame_wise(model, 7, nth_frame, img_size, optical_flow_dir)
measure_accuracy_frame_wise(model, 8, nth_frame, img_size, optical_flow_dir)
measure_accuracy_frame_wise(model, 9, nth_frame, img_size, optical_flow_dir)
measure_accuracy_frame_wise(model, 10, nth_frame, img_size, optical_flow_dir)


