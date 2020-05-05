from keras.models import load_model
import numpy as np
import cv2
from measure_kappa import query_img_names, get_mask

def getHandArea(frame, mask):
    new_frame = np.zeros_like(frame)
    new_frame[:,:,0] = np.bitwise_and(frame[:,:,0], mask[:,:,0])
    new_frame[:,:,1] = np.bitwise_and(frame[:,:,1], mask[:,:,0])
    new_frame[:,:,2] = np.bitwise_and(frame[:,:,2], mask[:,:,0])

    return new_frame

'''
path = 'check_points/UnetDataset_v0_UNet_01-0.91.hdf5'
model = load_model(path)

img_path = 'UnetDataset_v1/validation/HIJEL (107)-953-1.jpg'
frame = cv2.imread(img_path, 1)

resized = cv2.resize(frame, (512, 512), interpolation=cv2.INTER_LINEAR)
mask = get_mask(model, resized)
#img_1_1 = cv2.cvtColor(resized, cv2.cv2.COLOR_BGR2RGB)
hand_img = getHandArea(resized, mask)    
# Display the resulting frame
cv2.imshow('frame',hand_img)
cv2.waitKey()
'''
