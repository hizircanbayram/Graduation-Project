def manuel_control(segmented_parent, normal_parent):
    '''
    This function provides us an interface to check if the segmented images are created correctly. 
    If they dont, we have an option to move a pair of images out of ten to the UnetDatasetProcedure/again/(hand no_person) 
    '''
    from matplotlib import pyplot as plt
    import cv2
    import keyboard

    def getHandArea(frame, mask):
        frame[:,:,0] = cv2.bitwise_and(frame[:,:,0], mask[:,:,0])
        frame[:,:,1] = cv2.bitwise_and(frame[:,:,1], mask[:,:,0])
        frame[:,:,2] = cv2.bitwise_and(frame[:,:,2], mask[:,:,0])

        return frame

    def press(event):
        if event.key == '0' or event.key == '1' or event.key == '2' or event.key == '3' or event.key == '4' or \
           event.key == '5' or event.key == '6' or event.key == '7' or event.key == '8' or event.key == '9'    :
            
            segmented_path = objs[int(event.key)][0]
            parsed_segmented_path = segmented_path.split('/')
            segmented_name = parsed_segmented_path[len(parsed_segmented_path) - 1]
            person_dir = parsed_segmented_path[len(parsed_segmented_path) - 2]
            os.rename(segmented_path, 'UnetDatasetProcedure/again/' + person_dir + '/' + segmented_name)
            with open("errors.log", "a") as logfile:
                logfile.write('MANUEL CONTROL: ' + segmented_path + ' moved to ' + 'UnetDatasetProcedure/again/' + person_dir + '/' + segmented_name + '\n')

            normal_path = objs[int(event.key)][1]
            parsed_normal_path = normal_path.split('/')
            normal_name = parsed_normal_path[len(parsed_normal_path) - 1]
            person_dir = parsed_normal_path[len(parsed_normal_path) - 2]
            os.rename(normal_path, 'UnetDatasetProcedure/again/' + person_dir + '/' + normal_name)
            with open("errors.log", "a") as logfile:
                logfile.write('MANUEL CONTROL: ' + normal_path + ' moved to ' + 'UnetDatasetProcedure/again/' + person_dir + '/' + normal_name + '\n')            



    segmented_names = os.listdir(segmented_parent)
    normal_names = os.listdir(normal_parent)
    m = 0
    done = False

    while True:
        f, axarr = plt.subplots(2,5, figsize=(12.6,9.6))
        f.canvas.mpl_connect('key_press_event', press)
        
        objs = []
        for i in range(2):
            for k in range(5):
                if m >= len(normal_names):
                    done = True
                    break
                segmented_name = normal_names[m][:-4] + '-colormask.png'
                segmented_path = segmented_parent + '/' + segmented_name
                segmented_frame = cv2.imread(segmented_path)
                normal_path = normal_parent + '/' + normal_names[m]
                normal_frame = cv2.imread(normal_path)
                normal_frame = cv2.cvtColor(normal_frame, cv2.COLOR_RGB2BGR)
                combined_frame = getHandArea(normal_frame, segmented_frame)
                axarr[i][k].imshow(combined_frame)
                objs.append((segmented_path, normal_path))
                m += 1
            if i == 1 or m >= len(normal_names):
                print(m, '/', len(normal_names))
            if done:
                break
        #mng = plt.get_current_fig_manager()
        #mng.full_screen_toggle()
        f.show()
        plt.waitforbuttonpress()
        #input()
        plt.close()
        if done:
            break


