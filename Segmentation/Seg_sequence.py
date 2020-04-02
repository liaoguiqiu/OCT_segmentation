operatedir_one  = "../../saved_original_for_seg/496.jpg"

from analy import Save_signal_flag
import cv2
import math
import numpy as np
from median_filter_special import  myfilter
import pandas as pd
import os
import torch
import scipy.signal as signal
from scipy.stats.stats import pearsonr   
import random
from time import time
from  path_finding import PATH
from scipy.ndimage import gaussian_filter1d
from Seg_one import Seg_One_Frame
 
class Seg_sequence(object):
     def __init__(self):
         
        self.operate_dir =   "../../saved_original_for_seg/"
        self.savedir_path = "../../saved_processed_seg/"
        self.savedir_path2 = "../../saved_processed_seg2/"
        self.savedir_path3 = "../../saved_processed_seg3/"
        self.savedir_path4 = "../../saved_processed_seg4/"
        #saved_processed_thickness
        self.savedir_thick = "../../saved_processed_thickness/"

        self.self_check_path_create(self.savedir_path)
        self.self_check_path_create(self.savedir_path2)
        self.self_check_path_create(self.savedir_path3)
        self.self_check_path_create(self.savedir_path4)
        
        self.thickness1=[]
        self.thickness2=[]
        self.thickness3=[]
        self.path1_mean=[]
        self.path1 =[]
        self.fram_seger = Seg_One_Frame()
        self.Aligh_flag =True
     def self_check_path_create(self,directory):
        try:
            os.stat(directory)
        except:
            os.mkdir(directory)  
 
    #input image to calculate the fist aveline, for get the local minimal value calculation 
     def calculate_the_average_line(self,img):
        calculate_w = 5 # only calculate the front10 lines
        H,W = img.shape
        new = img[ :,5:5+calculate_w]
        ave_line=new.sum(axis=1)/calculate_w
        return ave_line
        #the  validation functionfor check the matrix and can also be used for validate the correction result
     def seg_sequence_process(self  ):
        read_sequence = os.listdir(self.operate_dir) # read all file name
        seqence_Len = len(read_sequence)    # get all file number 
        img_path = self.operate_dir +   "500.jpg"
        Img = cv2.imread(img_path)  #read the first one to get the image size
        #gray_video  =   cv2.cvtColor(video, cv2.COLOR_BGR2GRAY)
 
        H,W,D= Img.shape  #get size of image
 
 
        save_sequence_num = 0  # processing iteration initial 
 
 
        for sequence_num in range(seqence_Len):
        #for i in os.listdir("E:/estimagine/vs_project/PythonApplication_data_au/pic/"):
                #start from 500
                real_num = sequence_num+500
                img_path = self.operate_dir + str(real_num)+ ".jpg" # starting from 10
                Img = cv2.imread(img_path)
                SegImg,Sob,Image_with_bonud,Boundaries = self.fram_seger.seg_process(Img)
                depth1 = Boundaries[1]-Boundaries[0]
                depth2 = Boundaries[2]-Boundaries[1]
                depth3 = Boundaries[3]-Boundaries[2]
                path1 =  Boundaries[0]

                path1mean = np.mean(path1)
                self.path1_mean.append(path1mean)
                self.thickness1.append(depth1)
                self.thickness2.append(depth2)
                self.thickness3.append(depth3)
                self.path1.append(path1)
                if self.Aligh_flag == True:
                    cali_shift = -int(path1mean - self.path1_mean[0])
                    Sob = np.roll(Sob, cali_shift, axis = 0)
                    Image_with_bonud= np.roll(Image_with_bonud, cali_shift, axis = 0)
                    Hs,Ws = Image_with_bonud.shape
                    shifter = self.path1[0] -self.path1[0][0]
                    for iter in range(Ws):
                        lineshift= -int(shifter[iter] )
                        Image_with_bonud[:,iter] =np.roll( Image_with_bonud[:,iter] ,lineshift)
                # save the segmentation reult
                cv2.imwrite(self.savedir_path  + str(real_num) +".jpg", Img)
                cv2.imwrite(self.savedir_path2  + str(real_num) +".jpg", Sob)

                cv2.imwrite(self.savedir_path3  + str(real_num) +".jpg", Image_with_bonud)
                if sequence_num>3:
                    # change the list to imag array
                    H1  = len(self.thickness1)
                    W1  = len(self.thickness1[0])
                    thi_img1 = np.array(self.thickness1)/2
                    thi_img2 = np.array(self.thickness2)/2
                    thi_img3 = np.array(self.thickness3)/2

                    cv2.imwrite(self.savedir_thick  +   "1.jpg", thi_img1.astype(np.uint8))
                    cv2.imwrite(self.savedir_thick  +   "2.jpg", thi_img2.astype(np.uint8))
                    cv2.imwrite(self.savedir_thick  +   "3.jpg", thi_img3.astype(np.uint8))




                    

if __name__ == '__main__':
    video_seg  = Seg_sequence()
    video_seg.seg_sequence_process() 
     