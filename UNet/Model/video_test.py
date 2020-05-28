import numpy as np
import cv2

from keras.models import load_model

from measure_jaccard import query_img_names, get_mask, iou_loss_core
from predict import getHandArea
from unet20 import unet

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

cap = cv2.VideoCapture('HIJEL(32).mp4')

model_path = 'UnetDataset_v4/unet20.hdf5'
dependencies = {
    'iou_loss_core': iou_loss_core
}
model = load_model(model_path, custom_objects=dependencies)
#model = unet()
#print(model.summary())

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame, (512, 512), interpolation=cv2.INTER_LINEAR)
    resized = cv2.resize(frame_rgb, (512, 512), interpolation=cv2.INTER_LINEAR)
    #resized = img_to_array(load_img(dir_name + ID, color_mode='rgb', target_size=(512,512,3), interpolation='bilinear')).astype(int)
    mask = get_mask(model, resized)
    #img_1_1 = cv2.cvtColor(resized, cv2.cv2.COLOR_BGR2RGB)
    hand_img = getHandArea(frame_resized, mask)    
    # Display the resulting frame
    cv2.imshow('frame',hand_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

