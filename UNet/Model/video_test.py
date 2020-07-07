import numpy as np
import cv2

from keras.models import load_model

from measure_jaccard import query_img_names, get_mask, iou_loss_score
from predict import getHandArea
from unet20 import unet

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

import matplotlib.pyplot as plt


def getBB(points):
    #print(points)
    points = np.transpose(np.array(points))
    x1 = min(points[0])
    y1 = min(points[1])
    x2 = max(points[0])
    y2 = max(points[1])
    return (x1, y1), (x2, y2)

def getLameBB(p_main_1, p_main_2):
    center_x = (p_main_1[0] + p_main_2[0]) // 2
    center_y = (p_main_1[1] + p_main_2[1]) // 2

    top_half_edge = 128
    bottom_half_edge = 128
    right_half_edge = 128
    left_half_edge = 128

    central_bb_p1_x = center_x - left_half_edge
    central_bb_p1_y = center_y - top_half_edge
    central_bb_p2_x = center_x + right_half_edge
    central_bb_p2_y = center_y + bottom_half_edge

    central_bb_p1 = (central_bb_p1_x, central_bb_p1_y)
    central_bb_p2 = (central_bb_p2_x, central_bb_p2_y)

    return central_bb_p1, central_bb_p2


def getCentralBB(p_main_1, p_main_2, bigger_than_256=False):
    center_x = (p_main_1[0] + p_main_2[0]) // 2
    center_y = (p_main_1[1] + p_main_2[1]) // 2

    if bigger_than_256:
        square_edge = max(p_main_2[1] - p_main_1[1], p_main_2[0] - p_main_1[0]) 
        if square_edge % 2 == 1:
            square_edge += 1

        top_half_edge = square_edge // 2
        bottom_half_edge = square_edge // 2
        right_half_edge = square_edge // 2
        left_half_edge = square_edge // 2
    else:
        top_half_edge = 128
        bottom_half_edge = 128
        right_half_edge = 128
        left_half_edge = 128

    central_bb_p1_x = center_x - left_half_edge
    central_bb_p1_y = center_y - top_half_edge
    central_bb_p2_x = center_x + right_half_edge
    central_bb_p2_y = center_y + bottom_half_edge

    if central_bb_p1_x < 0:
        central_bb_p2_x += abs(central_bb_p1_x) 
        central_bb_p1_x = 0

    if central_bb_p1_y < 0:
        central_bb_p2_y += abs(central_bb_p1_y)
        central_bb_p1_y = 0

    if central_bb_p2_x > 511:
        central_bb_p1_x -= central_bb_p2_x - 511
        central_bb_p2_x = 511

    if central_bb_p2_y > 511:
        central_bb_p1_y -= central_bb_p2_y - 511
        central_bb_p2_y = 511


    central_bb_p1 = (central_bb_p1_x, central_bb_p1_y)
    central_bb_p2 = (central_bb_p2_x, central_bb_p2_y)

    return central_bb_p1, central_bb_p2



def getExtremeImg(img, cnt):
    # https://stackoverflow.com/questions/51000056/find-contour-and-boundary-to-obtain-points-inside-image-opencv-python
    leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
    rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
    topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
    bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])



    return leftmost, rightmost, topmost, bottommost



def drawExtremePoints(img, lm, rm, tm, bm, clr):
    cv2.circle(img, lm, 6, clr, -1)
    cv2.circle(img, rm, 6, clr, -1)
    cv2.circle(img, tm, 6, clr, -1)
    cv2.circle(img, bm, 6,clr, -1)



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
square_edges = []


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
    cropped_hand_img = cv2.resize(frame_bgr, (256,256))

    if len(contours) >= 1:
        rect1 = cv2.minAreaRect(contours[0])
        lm, rm, tm, bm = getExtremeImg(frame_bgr, contours[0])
        indices1 = [] # this is created to calculate the area of the first contour, which is the biggest area
        indices1 = addPOintsToTheList(indices1, lm, rm, tm, bm)
        p1_1, p1_2 = getBB(indices1)
        box1_area = (p1_2[0] - p1_1[0]) * (p1_2[1] - p1_1[1])
        if box1_area > 5000:
            drawExtremePoints(frame_bgr, lm, rm, tm, bm, (255,0,0))
            indices = addPOintsToTheList(indices, lm, rm, tm, bm) # adding points to the real list where the BB of the hands will be extracted
    if len(contours) > 1:
        rect2 = cv2.minAreaRect(contours[1])
        lm, rm, tm, bm = getExtremeImg(frame_bgr, contours[1])
        indices2 = [] # this is created to calculate the area of the first contour, which is the biggest area
        indices2 = addPOintsToTheList(indices2, lm, rm, tm, bm)
        p2_1, p2_2 = getBB(indices2)
        box2_area = (p2_2[0] - p2_1[0]) * (p2_2[1] - p2_1[1])
        if box2_area > 5000:
            indices = addPOintsToTheList(indices, lm, rm, tm, bm) # adding points to the real list where the BB of the hands will be extracted
            drawExtremePoints(frame_bgr, lm, rm, tm, bm, (0,255,0))
    if len(contours) >= 1 and len(indices) > 0: # one of them is contours,the other is  indices. dont confuse
        p_main_1, p_main_2 = getBB(indices)
        if (p_main_2[0] - p_main_1[0] > 256) or (p_main_2[1] - p_main_1[1] > 256):
            central_bb_p1, central_bb_p2 = getCentralBB(p_main_1, p_main_2, bigger_than_256=True)
            cropped_hand_img = frame_bgr[central_bb_p1[1] : central_bb_p2[1], central_bb_p1[0] : central_bb_p2[0]]
            cropped_hand_img = cv2.resize(cropped_hand_img, (256, 256), interpolation=cv2.INTER_LINEAR)
        else:
            central_bb_p1, central_bb_p2 = getCentralBB(p_main_1, p_main_2)
            cropped_hand_img = frame_bgr[central_bb_p1[1] : central_bb_p2[1], central_bb_p1[0] : central_bb_p2[0]]
        cv2.rectangle(frame_bgr, central_bb_p1, central_bb_p2, (255,255,255), 1) # white: square
        #lame_bb_p1, lame_bb_p2 = getLameBB(p_main_1, p_main_2)
        #cv2.rectangle(frame_bgr, lame_bb_p1, lame_bb_p2, (0,0,0), 2) # black: square
        cv2.rectangle(frame_bgr, p_main_1, p_main_2, (0,0,255), 1) # red: rectangle
        
    # Display the resulting frame
    cv2.imshow('frame', cropped_hand_img)
    #cv2.waitKey(0)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

plt.hist(np.array(square_edges))
plt.show()
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()