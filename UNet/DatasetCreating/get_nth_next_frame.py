'''
Bu dosya optik akis goruntulerinin cikarilmasi icin olusturulmustur. Veri kumemde sadece rastgele sectigim el goruntuleri vardi
Optik akis goruntuleri icin ise herhangi video karesinden bir, iki, ... ya da n adet sonraki video karelerine de ihtiyac var.
Bu noktada bu dosya mevcut veri kumemdeki goruntulerin isminden hareketle kac video karesi sonraki goruntuler isteniyorsa
o goruntulerin isimlerini names.txt dosyasina kaydeder(createFileNames fonksiyonu ile). Ardindan names.txt dosyasi tum videolarin
bulundugu klasore getirilir ve orada revertBackFileNames isimli fonksiyon calistirilir. Bu fonksiyon names.txt'de isimleri
bulunan goruntulerin createFileNames'e parametre olarak kadar gecilen sayi kadar sonraki video karelerini imgs isimli bir 
klasore koyar. 
createFileNames fonksiyonu UnetDataset_v4(yaklasik 10k goruntu iceren Unet veri kumem) kumesinin oldugu yerde cagrilirken
revertBackFileNames fonksiyonu videolarin bulundugu ve siniflandirma icin kullanilan en buyuk veri kumesinin orada cagrilir.
'''

def getImgName(frame_no_str):
    if len(frame_no_str) == 1:
        return "000" + frame_no_str
    elif len(frame_no_str) == 2:
        return "00" + frame_no_str
    elif len(frame_no_str) == 3:
        return "0" + frame_no_str
    elif len(frame_no_str) == 4:
        return frame_no_str
    else:
        print('SOMETHING WRONG')

        
def createFileNames(pixel_ahead_no):
    import os
    test_file_names = os.listdir('UnetDataset_v4/test')
    valid_file_names = os.listdir('UnetDataset_v4/validation')
    train_file_names = os.listdir('UnetDataset_v4/train')
    file_names = [ file_name for file_name in (test_file_names + valid_file_names + train_file_names) if file_name.endswith('.jpg') ]
    print(len(file_names))
    parsed_file_names = sorted([file_name.split('-')[0] + '.mp4/normal/' + getImgName(str(int(file_name.split('-')[1]) + pixel_ahead_no)) + '.png' + '-' + file_name.split('-')[2][0:len(file_name.split('-')[2]) - 4] for file_name in file_names])
    print(len(parsed_file_names))
    f = open("names.txt", "w")
    for p_name in parsed_file_names:
        f.write(p_name + '\n')


def getImgNameBack(frame_no_str):    
    if frame_no_str[0:4] == "0000":
        return frame_no_str[3:4]
    elif frame_no_str[0:3] == "000":
        return frame_no_str[3:4]
    elif frame_no_str[0:2] == "00":
        return frame_no_str[2:4]
    elif frame_no_str[0:1] == "0":
        return frame_no_str[1:4]
    elif len(frame_no_str) == 4:
        return frame_no_str
    else:
        print('SOMETHING WRONG', frame_no_str)
        
        
def revertBackFileNames():
    from shutil import copyfile
    f = open("names.txt", "r")
    file_names = f.readlines()

    print(len(file_names))
    for i, parsed_file_name in enumerate(file_names):
        split_name = parsed_file_name.split('/')
        movement_no = parsed_file_name.split('-')[1]
        movement_no = movement_no[0:len(movement_no) -1]  
        movement_no+= '.jpg'
        video_name = split_name[0]
        video_name = video_name[0:len(video_name) - 4]
        frame_no_str = split_name[2]
        frame_no_str = getImgNameBack(frame_no_str[0:4])   
        new_name = video_name + '-' + frame_no_str + '-' + movement_no
        #print(i)
        try:
            src_path = parsed_file_name.split('-')[0]
            copyfile(src_path[0:len(src_path)-4] + '.png', 'imgs/' + new_name)
        except:
            print(video_name, frame_no_str, movement_no)



createFileNames(2)
#revertBackFileNames()