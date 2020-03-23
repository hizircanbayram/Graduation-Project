import os

def getMovementNumbers(img_dir):
    '''
    This function returns the number of images per hand washing movement in an array 
    when given the path of directory that consists of hand washing frames.
    example return: [10,0,0,0,0,0,0,0,0,0,10] -> contains 10 frames from movement 1, 10 frames from movement 11
    Note: This function has to be used after the dataset, that will be used for training, is created with its training, validation and test subsets.
    '''
    img_names = os.listdir(img_dir)
    movement_nos = [0,0,0,0,0,0,0,0,0,0,0]
    for img_name in img_names:
        if img_name.endswith('.png'):
            splitted = img_name.split('-')
            index = int(splitted[len(splitted) - 2])
            movement_nos[index] += 1
    return movement_nos

main_dir_path = 'UnetDataset_v1/'

train_path = main_dir_path + 'train'
validation_path = main_dir_path + 'validation'
test_path = main_dir_path + 'test'

print('train: ', getMovementNumbers(train_path))
print('validation: ', getMovementNumbers(validation_path))
print('test: ', getMovementNumbers(test_path))
