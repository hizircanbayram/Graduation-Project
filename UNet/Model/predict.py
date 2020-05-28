from keras.models import load_model
import numpy as np



def getHandArea(frame, mask):
    new_frame = np.zeros_like(frame)
    new_frame[:,:,0] = np.bitwise_and(frame[:,:,0], mask[:,:,0])
    new_frame[:,:,1] = np.bitwise_and(frame[:,:,1], mask[:,:,0])
    new_frame[:,:,2] = np.bitwise_and(frame[:,:,2], mask[:,:,0])

    return new_frame


def get_mask(model, img):
    # BGR goruntu ile calisir
    img = img / 255.0
    img = img[np.newaxis, :, :, :]        
    segmented_pred = model.predict(img)
    segmented_pred = segmented_pred[0]
    segmented_pred = segmented_pred > 0.5
    segmented_pred = segmented_pred * 255

    return segmented_pred
