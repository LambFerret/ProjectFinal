from PIL import Image
import os
import re
root_dir = './Winter LandScape'
img_content = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]', '', root_dir)
print(img_content)
clean_path = './clean_'+img_content
img_path_list = []
img_extension = '.jpg'
file_number = 0
os.mkdir(clean_path)
for (root, dirs, files) in os.walk(root_dir):
    if len(files) > 0:
        for file_name in files:
            if os.path.splitext(file_name)[1] == img_extension:
                file_number += 1
                img_path = root + '/' + file_name
                img_path = img_path.replace('\\', '/')      
                image1 = Image.open(img_path)
                croppedimage = image1.crop((0,0,image1.size[0],260))
                croppedimage.save(clean_path +'/'+img_content+str(file_number)+'.jpg')
                
