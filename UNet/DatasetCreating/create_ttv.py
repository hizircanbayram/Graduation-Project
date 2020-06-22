

def create_train_test_validation_dataset(src_dir_paths, dataset_rates):
    import os
    from random import randrange
    from keras.preprocessing.image import load_img, img_to_array, save_img


    def copy_files(target_dir, dataset_no):
        turn_no = dataset_no
        for i in range(turn_no):
            indice = randrange(dataset_no)
            normal_image = img_to_array(load_img('normal' + '/' + src_dir_path + '/' + normal_image_names[indice], color_mode='rgb', target_size=(512,512), interpolation='bilinear'))
            save_img(target_dir + '/' + normal_image_names[indice], normal_image)

            segmented_image_name = normal_image_names[indice][:-4] + '-colormask.png'
            segmented_image = img_to_array(load_img('segmented' + '/' + src_dir_path + '/' + segmented_image_name, color_mode='grayscale', target_size=(512,512), interpolation='bilinear'))
            save_img(target_dir + '/' + segmented_image_name, segmented_image)

            normal_image_names.remove(normal_image_names[indice])
            dataset_no -= 1

        print(target_dir + ' is done')
        
    for src_dir_path in src_dir_paths:
        normal_image_names = os.listdir('normal' + '/' + src_dir_path)
        image_no = len(normal_image_names)

        indices = []
        for i in range(image_no):
            indices.append(i)
        train_no = int(image_no * dataset_rates[0])
        test_no = int(image_no * dataset_rates[1])
        validation_no = image_no - (train_no + test_no)

        copy_files('train', train_no)
        copy_files('test', test_no)
        copy_files('validation', validation_no)


    
        

create_train_test_validation_dataset(['0','1','2','3','4','5','6','7','8','9', '10'], [0.88, 0.06])



