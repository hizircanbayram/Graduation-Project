import numpy as np
import cv2
import imutils
import os
from random import randrange


def videoToNumpy(video_name, width, height, slidingSize, grayScale = True):
    '''
    Resizes the video given its name as parameter into width x height size, converts it into gray scale, 
    takes only multiple of slidingSize amount of the frames and puts it into a numpy array, order them in a way they can fit into to the model based on sliding size and returns it
    example : takes 979 frame, disards its 19 frame if the slidingSize is 40 so that 960 % 40 can be 0. If the image size is 128x128, 960x128x128 shape is created. Then 
    the function turns it into 24x40x128x128 so that each 40x128x128 sliding data can be predicted.
    Param | video_name : Name of the video whose content is converted into a numpy array
    Param | width : Width of the new frame 
    Param | height : Height of the new frame
    Param | slidingSize : Window sliding size which is determined during training so as to increase accuracy
    Return | numpy_vid : Content of the video whose name is passed as parameter in a numpy array format	
    This function is used in takeOneFrameFromEveryMovementInEveryVideo() function.		
    '''

    cap = cv2.VideoCapture(video_name)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #print('frame count : ', frameCount)
    numpy_frameCount = frameCount - (frameCount % slidingSize) # so as to fit the video into model, we need multiple of window sliding size amount of frame
    # For example if we have 979 frames and windowing size is 40, then we need to use 960 frames. The last 19 frames can be discarded
    if grayScale:
        numpy_vid_gray = np.empty((numpy_frameCount, height, width), np.dtype('uint8'))
    else:
        numpy_vid_gray = np.empty((numpy_frameCount, height, width, 3), np.dtype('uint8'))
    retVal = []
    fc = 0
    frameExists = True

    while fc < numpy_frameCount and frameExists:
        frameExists, frame = cap.read()
        if frameExists:
            if grayScale:
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                numpy_vid_gray[fc] = cv2.resize(frame_gray, (width, height))
            else:
                numpy_vid_gray[fc] = cv2.resize(frame, (width, height))
        else:
            if fc > 0:
                numpy_vid_gray[fc] = numpy_vid_gray[fc - 1]
            else:
                numpy_vid_gray[fc] = None
        fc += 1
    cap.release()

    return numpy_vid_gray





def getFrameIndexWithIntendedLabel(file_name, label_no):
    '''
    This function returns the first and last index of any hand movement determined by label_no in the xml file determined by file_name.
    This function is used in takeOneFrameFromEveryMovementInEveryVideo() function.
    '''
    import xml.etree.ElementTree as ET
    tree = ET.parse(file_name)
    root = tree.getroot()
    first_encountered = False
    last_encountered = False
    first = -1
    last = -1
    for i in range(len(root)):
        if (i - 1) >= 0 and root[i][1].text == str(label_no) and root[i - 1][1].text != str(label_no):
            first = i + 1
            first_encountered = True
        elif (i + 1) < len(root) and root[i][1].text == str(label_no) and root[i + 1][1].text != str(label_no):
            last = i + 1
            last_encountered = True
        if (first_encountered == True) and (last_encountered == True):
            return first, last
    if first == -1 or last == -1:
        print('getFrameIndexWithIntendedLabel: Undefined label given as parameter, returning -1. Given label_no: ', label_no)
    return first, last



def takeOneFrameFromEveryMovementInEveryVideo(main_dir='dataset_hij', target_dir='video_dataset'):
    '''
    Creates video dataset by taking a frame from every hand movement in any video, when possible. 
    '''
    dir_names = os.listdir(main_dir)
    os.mkdir(target_dir)
    frame_numbers = [0,0,0,0,0,0,0,0,0,0,0,0]
    with open('data_info_mass.log', 'r') as dim:
        for k, line in enumerate(dim):
            toks = line.split('**')
            dir_name = toks[0]
            video_name = toks[1]
            xml_name = toks[2]
            
            new_dir_name = target_dir + '/' + video_name
            os.mkdir(new_dir_name)
            frames = videoToNumpy(main_dir + '/' + video_name + '/' + video_name, 512, 512, 1, grayScale=False)
            indexes = []
            for i in range(1, 12):
                first, last = getFrameIndexWithIntendedLabel(main_dir + '/' + video_name + '/' + xml_name, i)
                if first != -1 and last != -1:
                        frame_numbers[i] += 1
                        random_index = randrange(first, last)
                        indexes.append(random_index)
                for index in indexes:
                    cv2.imwrite(target_dir + '/' + video_name + '/' + video_name + '_' + str(index) + '.jpg',frames[index])
            print(k, '/975 ', frame_numbers)



def move_xml_files_to(main_dir, target_dir):
    '''
    This function move xml files from main_dir to target_dir
    '''
    import os
    from shutil import copyfile

    dir_names = os.listdir(main_dir)

    for dir_name in dir_names:
        xml_name = None
        if dir_name == '.' or dir_name == '..':
            continue
        file_names = os.listdir(main_dir + '/' + dir_name)
        for file_name in file_names:
            elif file_name.endswith('.xml'):
                xml_name = file_name
        copyfile(main_dir + '/' + dir_name + '/' + xml_name, target_dir + '/' + dir_name + '/' + xml_name)



move_xml_files_to('dataset_hij', 'class_dataset')
takeOneFrameFromEveryMovementInEveryVideo()
