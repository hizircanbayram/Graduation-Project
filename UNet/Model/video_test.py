import numpy as np
import cv2

from keras.models import load_model

from measure_jaccard import query_img_names, get_mask, iou_loss_core
from predict import getHandArea
from unet20 import unet

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array


def getBB(points):
    #print(points)
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
my_key = False
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
    #indices = []
    rect1 = cv2.minAreaRect(contours[0])
    box1 = cv2.boxPoints(rect1)
    box1 = np.int0(box1)
    p3_1, p3_2 = getBB(box1)
    #cv2.rectangle(frame_resized, p3_1, p3_2, (0,0,255), 2)   
    indices = None
    if len(contours) > 1:
        rect2 = cv2.minAreaRect(contours[1])
        box2 = cv2.boxPoints(rect2)
        box2 = np.int0(box2)
        p2_1, p2_2 = getBB(box2)
        box2_area = (p2_2[0] - p2_1[0]) * (p2_2[1] - p2_1[1])
        #print('cons')
        if box2_area > 1000:
            #print('> 100')
            indices = box1.tolist() + box2.tolist()
            #print(type(box1))
            #print('b1', box1, 'b2', box2, 'b3', indices)
            #cv2.waitKey(0)
        else:
            indices = box1.tolist()   
    else:
        #print('else')
        indices = box1.tolist()

    p1_1, p1_2 = getBB(indices)
        
    
    if box2_area >= 1000:
        #print('contour 2, area: ', box2_area)
        #cv2.rectangle(frame_resized, p2_1, p2_2, (0,255,0), 2)
        my_key = True
    cv2.rectangle(frame_resized, p1_1, p1_2, (255,0,0), 2)

    # Display the resulting frame
    cv2.imshow('frame',frame_resized)
    #if my_key:
    #    cv2.waitKey(0)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    my_key = False
  
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
