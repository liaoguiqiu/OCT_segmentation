dir_thickmap  = "../../saved_processed_thickness/"
operatedir_pic =  "..\\rotate\\"
savedir_xml  =   "..\\add_xml\\"
savedir_pic  =   "..\\add\\"
backgrounddir=   "..\\background\\"
 
#翻转图片的数据集增强
import cv2
import math
import numpy as np
import os
import random
class Post_poss(object):
    def __init__(self ):
        #self. dir_thickmap  = "../../saved_processed_thickness/"
        self.savedir_path2 = "../../saved_processed_seg2/"
        self.savedir_path3 = "../../saved_processed_seg3/"
        self.savedir_path4 = "../../saved_processed_seg4/"
        self.savedir_path2_Sele = "../../saved_processed_seg2_sele/"
        #saved_processed_thickness
        self.savedir_thick = "../../saved_processed_thickness/"

        self.self_check_path_create(self.savedir_thick)
        self.self_check_path_create(self.savedir_path2)
        self.self_check_path_create(self.savedir_path3)
        self.self_check_path_create(self.savedir_path4)
        self.self_check_path_create(self.savedir_path2_Sele)

    def self_check_path_create(self,directory):
        try:
            os.stat(directory)
        except:
            os.mkdir(directory)  
    def delete_the_error_heat(self):
        heat1  = cv2.imread(self.savedir_thick +"1.jpg")
        heat2  = cv2.imread(self.savedir_thick +"2.jpg")
        heat3  = cv2.imread(self.savedir_thick +"3.jpg")
        heat1  =   cv2.cvtColor(heat1, cv2.COLOR_BGR2GRAY)
        heat2  =   cv2.cvtColor(heat2, cv2.COLOR_BGR2GRAY)
        heat3  =   cv2.cvtColor(heat3, cv2.COLOR_BGR2GRAY)
        new_h = len(os.listdir(self.savedir_path4))
        H,W = heat1.shape

        mask =   heat1*0 
        New1= np.zeros((H,W))
        New2= np.zeros((H,W))
        New3= np.zeros((H,W))
        for i in range(201):
            img_path = self.savedir_path4 + str(i+500) + ".jpg"
            img = cv2.imread(img_path)
            if img is None:
                mask[i,:] = mask[i,:] +255
                print ("no_img for this  " +str(i+500))
                #remove the heat


            else:
                selepath =  self.savedir_path3 + str(i+500) + ".jpg"
                img_se = cv2.imread(selepath)
                cv2.imwrite(self.savedir_path2_Sele  + str(i+500)+  ".jpg", img_se)
        New1 = cv2.inpaint(heat1, mask, 3, cv2.INPAINT_TELEA)
        New2 = cv2.inpaint(heat2, mask, 3, cv2.INPAINT_TELEA)

        New3 = cv2.inpaint(heat3, mask, 3, cv2.INPAINT_TELEA)
        cv2.imwrite(self.savedir_thick  +   "n1.jpg", New1.astype(np.uint8))
        cv2.imwrite(self.savedir_thick  +   "n2.jpg", New2.astype(np.uint8))
        cv2.imwrite(self.savedir_thick  +   "n3.jpg", New3.astype(np.uint8))
        #for i in os.listdir(self.savedir_path3):
        ##for i in os.listdir("E:\\estimagine\\vs_project\\PythonApplication_data_au\\pic\\"):
        #    # 分离文件名与后缀
        #    a, b = os.path.splitext(i)
        #    # 如果后缀名是“.xml”就旋转related的图像
        #    if b == ".jpg":
        #        img_path = self.savedir_path3 + a + ".jpg"
        #        img1 = cv2.imread(img_path)
if __name__ == '__main__':
     process  = Post_poss()
     process.delete_the_error_heat() 