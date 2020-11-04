# update 4:38 2nd OCt 2020
# this is to modify/opy a exiting Json file to generate the contour of the theatht 
#!!! this auto json is to generate json for images without label file , this willl generate a lot of json file 
import json as JSON
import cv2
import math
import numpy as np
import os
import random 
from zipfile import ZipFile
import scipy.signal as signal
import pandas as pd
from generator_contour import Save_Contour_pkl
from seg_one_1 import Seg_One_1
import codecs
import PIL.ExifTags
import PIL.Image
import PIL.ImageOps
import base64
import io

def encodeImageForJson(image):
    img_pil = PIL.Image.fromarray(image, mode='RGB')
    f = io.BytesIO()
    img_pil.save(f, format='PNG')
    data = f.getvalue()
    encData = codecs.encode(data, 'base64').decode()
    encData = encData.replace('\n', '')
    return encData
class  Auto_json_label(object):
    def __init__(self ):
        #self.image_dir   = "../../OCT/beam_scanning/Data set/pic/NORMAL-BACKSIDE-center/"
        #self.roi_dir =  "../../OCT/beam_scanning/Data set/seg label/NORMAL-BACKSIDE-center/"
        #self.database_root = "../../OCT/beam_scanning/Data Set Reorganize/NORMAL/"
        #self.database_root = "../../OCT/beam_scanning/Data Set Reorganize/NORMAL-BACKSIDE-center/"
        #self.database_root = "../../OCT/beam_scanning/Data Set Reorganize/NORMAL-BACKSIDE/"
        jason_tmp_dir  =  "D:/Deep learning/dataset/original/phantom/1/label/0.json"
        # read th jso fie in hte start :
        with open(jason_tmp_dir) as dir:
            self.jason_tmp = JSON.load(dir)
        self.shapeTmp  = self.jason_tmp["shapes"]
        self.coordinates0 = self.jason_tmp["shapes"] [1]["points"] # remember add finding corred label 1!!!
        self.co_len = len (self.coordinates0) 
        self.f_downsample_factor  = 4
        self.database_root = "D:/Deep learning/dataset/original/phantom/2/"
        #self.database_root = "D:/Deep learning/dataset/original/phantom/1/"
        #self.database_root = "D:/Deep learning/dataset/original/animal_tissue/1/"
        #self.database_root = "D:/Deep learning/dataset/original/IVOCT/1/"
        self.image_dir   = self.database_root + "pic/"
        self.json_dir =  self.database_root + "label/" # for this class sthis dir ist save the modified json 
        self.json_save_dir  = self.database_root + "label_generate/"
        self.all_dir = self.database_root + "pic_all/"
        self.img_num = 0
         
        self.contours_x =  [] # no predefines # predefine there are 4 contours
        self.contours_y =  [] # predefine there are 4 contours
        self.seger = Seg_One_1()
        self.saver = Save_Contour_pkl()
        self.display_flag = True
    def draw_coordinates_color(self,img1,vx,vy,color):
        
        if color ==0:
           painter  = [254,0,0]
        elif color ==1:
           painter  = [0,254,0]
        elif color ==2:
           painter  = [0,0,254]
        else :
           painter  = [0,0,0]
                    #path0  = signal.resample(path0, W)
        H,W,_ = img1.shape
        for j in range (len(vx)):
                #path0l[path0x[j]]
                dy = np.clip(vy[j],2,H-2)
                dx = np.clip(vx[j],2,W-2)

                img1[int(dy)+1,int(dx),:]=img1[int(dy),int(dx),:]=painter
                #img1[int(dy)+1,dx,:]=img1[int(dy)-1,dx,:]=img1[int(dy),dx,:]=painter


        return img1
        #this is to down sample the image in one folder
    def downsample_folder(self):#this is to down sample the image in one folder
        read_sequence = os.listdir(self.all_dir) # read all file name
        seqence_Len = len(read_sequence)    # get all file number 
          
        for sequence_num in range(0,seqence_Len):
        #for i in os.listdir("E:/estimagine/vs_project/PythonApplication_data_au/pic/"):
            if (sequence_num%self.f_downsample_factor == 0):
                img_path = self.all_dir + str(sequence_num) + ".jpg"
                #jason_path  = self.json_dir + a + ".json"
                img1 = cv2.imread(img_path)
                
                if img1 is None:
                    print ("no_img")
                else:
                    # write this one into foler
                    cv2.imwrite(self.image_dir  + str(sequence_num) +".jpg",img1 )
                    print("write  " + str(sequence_num))
                    pass

            sequence_num+=1
            pass

    def check_one_folder (self):
        #for i in os.listdir(self.image_dir): # star from the image folder
        for i in os.listdir(self.image_dir): # star from the image folder

    #for i in os.listdir("E:\\estimagine\\vs_project\\PythonApplication_data_au\\pic\\"):
        # separath  the name of json 
            a, b = os.path.splitext(i)
            # if it is a img it will have corresponding image 
            if b == ".jpg" :
                img_path = self.image_dir + a + ".jpg"
                #jason_path  = self.json_dir + a + ".json"
                img1 = cv2.imread(img_path)
                
                if img1 is None:
                    print ("no_img")
                else:
                    gray  =   cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
                    H,W   = gray.shape
                    # thsi json dir will be used to save  the generated json
                    save_json_dir = self.json_save_dir + a + ".json"
                    #copy the temp json
                    #with open(json_dir) as dir:
                    #        this_json= JSON.load(dir)
                    this_json = self.jason_tmp
                    this_coodinates = self.coordinates0
                    this_shape = self.shapeTmp
                    shape_temp = self.shapeTmp
                    # the start should be choose larger than 50 , here it is 100
                    sheath_contour  = self.seger.seg_process(img1,100)
                    for iter  in range(2): #  2 contours here : iter is tje contoru index
                        if  shape_temp[iter]["label"]  =="1":
                            this_coodinates = shape_temp[iter]["points"] # remember add finding corred label 1!!!
                            co_len = len (this_coodinates) 
                            for iter2 in range (co_len):
                                this_px  = this_coodinates[iter2][0] 
                                this_coodinates[iter2][1] = sheath_contour[int(this_px)]
                            shape_temp[iter] ["points"] = this_coodinates   #modify the shape temp
                        else : 
                            for iter2 in range (len(this_shape)):
                                if  shape_temp[iter]["label"] == this_shape[iter2]["label"]:
                                    shape_temp[iter] ["points"] = this_shape[iter2] ["points"]

                    # modify the imag name and the height and width next time

                    
                    #for iter in range (self.co_len):
                    #    this_px  = this_coodinates[iter][0] 
                    #    this_coodinates[iter][1] = sheath_contour[int(this_px)]

                    this_json ["shapes"]   = shape_temp
                    this_json ["imageHeight"] = H
                    this_json ["imageWidth"] = W
                    this_json ["imagePath"] = a+ ".jpg"
                    this_json [ "imageData"]  = encodeImageForJson(img1)
                    #shape  = data["shapes"]
                    with open(save_json_dir, "w") as jsonFile:
                        JSON.dump(this_json, jsonFile)
                    #num_line  = len(shape)
                    #len_list=  num_line
                    #with open(json_dir) as f_dir:
                    #    data = JSON.load(f_dir)
if __name__ == '__main__':
        cheker  = Auto_json_label()
        cheker.downsample_folder()
        cheker.check_one_folder() 