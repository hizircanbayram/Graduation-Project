import numpy as np
import cv2

from keras.models import load_model

from measure_jaccard import query_img_names, get_mask, iou_loss_core
from predict import getHandArea
from unet20 import unet

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array


def getBB(points):
    points = np.transpose(np.array(points))
    x1 = min(points[0])
    y1 = min(points[1])
    x2 = max(points[0])
    y2 = max(points[1])
    return (x1, y1), (x2, y2)


cap = cv2.VideoCapture('HIJEL(32).mp4')

model_path = 'check_points/unet20_52_256/Unet20_07-0.96.hdf5'
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
    frame_resized = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_LINEAR)
    resized = cv2.resize(frame_rgb, (256, 256), interpolation=cv2.INTER_LINEAR)
    mask = get_mask(model, resized, 0)
    contours, hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    hand_img = getHandArea(frame_resized, mask)   
    #for c in contours:
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    rect = cv2.minAreaRect(contours[0])
    indices = []
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    p1, p2 = getBB(box)
    cv2.rectangle(frame_resized, p1, p2, (255,0,0), 2)

    # Display the resulting frame
    cv2.imshow('frame',frame_resized)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

  
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
