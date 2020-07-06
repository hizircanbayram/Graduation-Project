import numpy as np
import cv2

from keras.models import load_model

from measure_jaccard import query_img_names, get_mask, iou_loss_score
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



def getExtremeImg(im2, cnt):
    leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
    rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
    topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
    bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])

    #--- drawing these points on the image ----
    cv2.circle(im2, leftmost, 6, (0, 0, 255), -1)
    cv2.circle(im2, rightmost, 6, (0, 0, 255), -1)
    cv2.circle(im2, topmost, 6, (0, 255, 0), -1)
    cv2.circle(im2, bottommost, 6, (0, 255, 0), -1)

    return leftmost, rightmost, topmost, bottommost



def makeBlackBoundriesOfImage(mask):
    mask[:,len(mask[0,:,0])-1,0] = 0
    mask[len(mask[0,:,0])-1,:,0] = 0
    mask[0,:,0] = 0
    mask[:,0,0] = 0



def addPOintsToTheList(lst, lm, rm, tm, bm):
    lst.append(lm)
    lst.append(rm)
    lst.append(tm)
    lst.append(bm)
    return lst


cap = cv2.VideoCapture('HIJEL(32).mp4')


dependencies = {
    'iou_loss_score': iou_loss_score
}
model = load_model('check_points/unet20_50_512/5_temmuz_unet20_013_50.hdf5', custom_objects=dependencies)



while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_bgr = cv2.resize(frame, (512, 512), interpolation=cv2.INTER_LINEAR)
    resized = cv2.resize(frame_rgb, (512, 512), interpolation=cv2.INTER_LINEAR)

    mask = get_mask(model, resized, 0)
    makeBlackBoundriesOfImage(mask)

    contours, hierarchy = cv2.findContours(mask,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    indices = []
    box1_area = None
    box2_area = None

    if len(contours) >= 1:
        rect1 = cv2.minAreaRect(contours[0])
        lm, rm, tm, bm = getExtremeImg(frame_bgr, contours[0])
        indices1 = [] # this is created to calculate the area of the first contour, which is the biggest area
        indices1 = addPOintsToTheList(indices1, lm, rm, tm, bm)
        p1_1, p1_2 = getBB(indices1)
        box1_area = (p1_2[0] - p1_1[0]) * (p1_2[1] - p1_1[1])
        if box1_area > 1000:
            indices = addPOintsToTheList(indices, lm, rm, tm, bm) # adding points to the real list where the BB of the hands will be extracted
    if len(contours) > 1:
        rect2 = cv2.minAreaRect(contours[1])
        lm, rm, tm, bm = getExtremeImg(frame_bgr, contours[1])
        indices2 = [] # this is created to calculate the area of the first contour, which is the biggest area
        indices2 = addPOintsToTheList(indices2, lm, rm, tm, bm)
        p2_1, p2_2 = getBB(indices2)
        box2_area = (p2_2[0] - p2_1[0]) * (p2_2[1] - p2_1[1])
        if box2_area > 1000:
            indices = addPOintsToTheList(indices, lm, rm, tm, bm) # adding points to the real list where the BB of the hands will be extracted
    if len(contours) >= 1 and len(indices) > 0:
        p_main_1, p_main_2 = getBB(indices)
        cv2.rectangle(frame_bgr, p_main_1, p_main_2, (0,0,255), 2) # kirmizi: iyi

    # Display the resulting frame
    cv2.imshow('frame',frame_bgr)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()