import numpy as np
import cv2

from keras.models import load_model

from measure_jaccard import query_img_names, get_mask
from predict import getHandArea
from unet import unet

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from unet import iou_loss_core

import matplotlib.pyplot as plt
'''
img_to_array : default float32 olarak olusturur numpy array'i
plt.imshow() fonksiyonunun goruntu renderlamasi isteniyorsa
    rgb goruntuler icin parametre olarak gecilen numpy dizisi uint8 ya da float32 olmalidir
    grayscale goruntuler icin plt.imshow'a cmap='gray' parametresi gecilmelidir    
'''
f, axarr = plt.subplots(2,2, figsize=(12.6,9.6))
img11 = img_to_array(load_img('UnetDataset_v1/validation/HIJEL (29)-327-8-colormask.png', color_mode='grayscale', target_size=(256,256), interpolation='bilinear'), dtype='uint8')[:,:,0]
img12 = img_to_array(load_img('UnetDataset_v1/validation/HIJEL (2)-1994-10.jpg', color_mode='rgb', interpolation='bilinear'), dtype='uint8')
img2 = cv2.cvtColor(cv2.imread('UnetDataset_v1/validation/HIJEL (2)-1994-10.jpg', 1), cv2.COLOR_BGR2RGB)
img3 = plt.imread('UnetDataset_v1/validation/HIJEL (2)-1994-10.jpg')
#print(img11)
print('pix: ', img11[2,0])
print('pix: ', img11[10,10])
print('pix: ', img11[20,20])
print('pix: ', img11[30,30])
print('pix: ', img11[40,40])
print('pix: ', img11[50,50])
print('pix: ', img11[60,60])
print('pix: ', img11[70,70])
print('pix: ', img11[80,80])

print(img12)
'''
for i in range(img1.shape[0]):
    for k in range(img1.shape[1]):
        if (img1[i][k][0] != img2[i][k][0]):
            print(i, k, img1[i][k][0], img2[i][k][0])
        if (img1[i][k][1] != img2[i][k][1]):
            print(i, k, img1[i][k][1], img2[i][k][1])
        if (img1[i][k][2] != img2[i][k][2]):
            print(i, k, img1[i][k][2], img2[i][k][2])
        print('----- ----- ----- ----- -----')
'''

axarr[0][0].imshow(img11, cmap='gray')
axarr[0][1].imshow(img2)
f.show()
plt.waitforbuttonpress()
