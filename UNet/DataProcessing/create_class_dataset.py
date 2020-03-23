
def createClassDatasetFromVideoDataset(main_dir='video_dataset', target_dir='class_dataset'):
    '''
    This function creates a class dataset from the video dataset. It takes every image in every video directory in video dataset and relocated them based on their hand movement no.
    You need to create directories named 0, 1, 2, ..., 10 before invoking this function. 
    '''
    import os
    from shutil import copyfile

    def getLabelIndexFromFrameIndex(xml_name, file_name):
        import xml.etree.ElementTree as ET
        tree = ET.parse(xml_name)
        root = tree.getroot()
        name_parsed = file_name[:-4].split('_')
        frame_no = int(name_parsed[len(name_parsed) - 1])    
        label_no = int(root[frame_no][1].text)
        return label_no

    dir_names = os.listdir(main_dir)

    for dir_name in dir_names:
        label_no = None
        xml_name = None
        if dir_name == '.' or dir_name == '..':
            continue
        file_names = os.listdir(main_dir + '/' + dir_name)
        for file_name in file_names:
            if file_name.endswith('.xml'):
                xml_name = file_name        
        for file_name in file_names:
            if file_name.endswith('.jpg'):
                label_no = getLabelIndexFromFrameIndex(main_dir + '/' + dir_name + '/' + xml_name, main_dir + '/' + dir_name + '/' + file_name)
                copyfile(main_dir + '/' + dir_name + '/' + file_name, target_dir + '/' + str(label_no) + '/' + file_name)



def checkFramesInCorrectDirectory(target_dir='class_dataset', main_dir='video_dataset'):
    '''
    This function checks if there is any misplaced images in any directory in class_dataset.
    '''
    import os
    from unet_dataset_adjuster import getLabelIndexFromFrameIndex

    class_dirs = os.listdir(target_dir)
    for class_dir in class_dirs:
        label_no = int(class_dir)
        img_names = os.listdir(target_dir + '/' + class_dir)
        for img_name in img_names:
            name_tokens = img_name.split('_')
            k = len(img_name) - 1
            excess = ''
            while k >= 0:
                if img_name[k] != '_':
                    excess += img_name[k]
                else:
                    excess += '_'
                    break
                k -= 1
            main_dir_elm = main_dir + '/' + img_name[:-len(excess)]
            main_dir_elm_samples = os.listdir(main_dir_elm)
            xml_file_name = None
            for main_dir_elm_sample in main_dir_elm_samples:
                if main_dir_elm_sample.endswith('.xml'):
                    xml_file_name = main_dir_elm_sample
            if getLabelIndexFromFrameIndex(main_dir_elm + '/' + xml_file_name, img_name) != label_no:
                print(img_name + ' does not belong to ' + str(label_no))
        print(str(label_no) + ' is done')



createClassDatasetFromVideoDataset()
checkFramesInCorrectDirectory()
