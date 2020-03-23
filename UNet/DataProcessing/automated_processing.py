def check_pixels(dir_path):
    '''
    Given a directory path, this function checks if the image created by pixel annotation tool contains pixel values different than either 0 or 255.
    If it does, the image is moved to again/(directory name the image belongs to) directory and log file is updated.
    If it does not, the image is moved to success/(directory name the image belongs to) directory.
    '''
    import os
    import cv2
    def check_pixels_for_img(img_path):
        img = cv2.imread(img_path)
        parsed_dir_path = img_path.split('/')
        img_name = parsed_dir_path[len(parsed_dir_path) - 1]
        dir_name = parsed_dir_path[len(parsed_dir_path) - 2]
        smt_wrong = False
        if 'colormask' not in img_name:
            return
        for i in range(img.shape[0]):
            for k in range(img.shape[1]):
                if img[i][k][0] == 250:
                    img[i][k][0] = 255
                if img[i][k][1] == 250:
                    img[i][k][1] = 255
                if img[i][k][2] == 250:
                    img[i][k][2] = 255

                if  ((    (img[i][k][0] != 0) and (img[i][k][0] != 255)) or \
                     (    (img[i][k][1] != 0) and (img[i][k][1] != 255)) or \
                     (    (img[i][k][2] != 0) and (img[i][k][2] != 255))):
                    smt_wrong = True
                    os.rename(img_path, parent_dir + '/' + 'again' + '/' + dir_name + '/' + img_name)
                    os.rename(img_path[:-14] + '.jpg', parent_dir + '/' + 'again' + '/' + dir_name + '/' + img_name[:-14] + '.jpg')                    
                    with open("errors.log", "a") as logfile:
                        logfile.write('CHECK PIXELS: ' + img_name + ' has pixel intensity values different than either 0 or 255: ' + str(img[i][k]) + '\n')
                    break
            if smt_wrong:
                break
        if not smt_wrong:
            os.rename(img_path, parent_dir + '/' + 'success' + '/' + dir_name + '/' + img_name)         

    print(dir_path)
    file_names = os.listdir(dir_path)
    for i, file_name in enumerate(file_names):
        check_pixels_for_img(dir_path + '/'  + file_name)
        if (i != 0) and (i % 100 == 0):
            print(i, ' imgs are checked about their pixel values')



def move_images_to_the_dataset(dir_name):
    import os
    '''
    Given a directory path this function moves images from UnetDatasetProcedure/name_(hand no) and UnetDatasetProcedure/success/name_(hand no) to
    UnetDataset/normal/name_(hand no) and UnetDataset/segmented/name_(hand no) 
    '''
    src = 'UnetDatasetProcedure/' + dir_name
    dest_seg = 'UnetDataset/segmented/' + dir_name
    dest_nor = 'UnetDataset/normal/' + dir_name
    src_file_names = os.listdir(src)
    for file_name in src_file_names:
        if file_name.endswith('.jpg'): # normal
            os.rename('UnetDatasetProcedure/' + dir_name + '/' + file_name, dest_nor + '/' + file_name)
            os.rename('UnetDatasetProcedure/' + 'success/' + dir_name + '/' + file_name[:-4] + '-colormask.png', dest_seg + '/' + file_name[:-4] + '-colormask.png')



def check_corresponding_images(dir_name):
    '''
    Given directory name(not path), this function checks if the images in either directory normal or segmented, have their corresponding images in the other directory.
    If they don't, the images are deleted so that only images that have their corresponding images exist 
    '''
    import os
    norm_path = 'UnetDataset/normal/' + dir_name
    segm_path = 'UnetDataset/segmented/' + dir_name

    norm_img_names = os.listdir(norm_path)
    segm_img_names = os.listdir(segm_path)

    counter = 0
    if len(norm_img_names) > len(segm_img_names):
        img_names = norm_img_names
        for img_name in img_names:
            if not os.path.exists(segm_path + '/' + img_name[:-4] + '-colormask.png'):
                missing_file_path = norm_path + '/' + img_name
                os.remove(missing_file_path)
                print(missing_file_path + ' is deleted')
    else:
        img_names = segm_img_names
        for img_name in img_names:
            if not os.path.exists(norm_path + '/' + img_name[:-14] + '.jpg'):
                missing_file_path = segm_path + '/' + img_name
                os.remove(missing_file_path)
                print(missing_file_path + 'is deleted')




parent_dir = 'UnetDatasetProcedure'

dir_name = '0_mustafa'
#check_pixels(parent_dir + '/' + dir_name)
#move_images_to_the_dataset(dir_name)
#check_corresponding_images(dir_name)

