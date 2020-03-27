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

        self.self_check_path_create(self.savedir_path)
        self.self_check_path_create(self.savedir_path2)
        self.self_check_path_create(self.savedir_path3)


        self.fram_seger = Seg_One_Frame()
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
                SegImg,Sob,Boundaries = self.fram_seger.seg_process(Img)

                # save the segmentation reult
                cv2.imwrite(self.savedir_path  + str(real_num) +".jpg", Img)
                cv2.imwrite(self.savedir_path2  + str(real_num) +".jpg", Sob)

                cv2.imwrite(self.savedir_path3  + str(real_num) +".jpg", Boundaries)

                    

if __name__ == '__main__':
    video_seg  = Seg_sequence()
    video_seg.seg_sequence_process() 
     