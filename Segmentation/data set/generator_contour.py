import cv2
import math
import numpy as np
import os
import random
from matplotlib.pyplot import *
#from mpl_toolkits.mplot3d import Axes3D
from median_filter_special import  myfilter
from cost_matrix import  COSTMtrix
from scipy.ndimage import gaussian_filter1d

#PythonETpackage for xml file edition
try: 
    import xml.etree.cElementTree as ET 
except ImportError: 
    import xml.etree.ElementTree as ET 
import sys 
#GPU acceleration
from cost_matrix import Window_LEN
from analy_visdom import VisdomLinePlotter
#from numba import vectorize
#from numba import jit
import pickle
from enum import Enum
import scipy.signal as signal
from operater import Basic_Operator
import math
#from ImgJ_ROI2 import Read_read_check_ROI_label
class Save_Contour_pkl(object):
    def __init__(self ):
        #set = Read_read_check_ROI_label()
        #self.database_root = set.database_root
        #check or create this path
        #self.self_check_path_create(self.signal_data_path)
        self.img_num= []
        self.contoursx = []
        self.contoursy = []
    def self_check_path_create(self,directory):
        try:
            os.stat(directory)
        except:
            os.mkdir(directory)  
   # add new step of all signals
    def append_new_name_contour(self,number,this_contoursx,this_contoursy,dir):
        #buffer
        self.img_num.append(number)
        self.contoursx.append(this_contoursx)
        self.contoursy.append(this_contoursy)

        #save the data 
        save_path = dir + "seg label pkl/"
        with open(save_path+'contours.pkl', 'wb') as f:
            pickle.dump(self , f, pickle.HIGHEST_PROTOCOL)
        pass
    #read from file
    def read_data(self,base_root):
        saved_path  = base_root  + "seg label pkl/"
        self = pickle.load(open(saved_path+'contours.pkl','rb'),encoding='iso-8859-1')
        return self

class Generator_Contour(object):
    def __init__(self ):
        self.origin_data = Save_Contour_pkl()
        self.database_root = "../../OCT/beam_scanning/Data Set Reorganize/VARY/"
        #self.database_root ="../../OCT/beam_scanning/Data Set Reorganize/NORMAL/"
        self.origin_data =self.origin_data.read_data(self.database_root)
        self.back_ground_root  =  "../../"     + "saved_background_for_generator/"

        self.save_img_dir = "../../"     + "saved_generated_contour/"
        self.save_contour_dir = "../../"     + "saved_stastics_coutour_generated/"
        self.display_flag =True
        self.img_num= []
        self.contoursx = []
        self.contoursy = []
        #check or create this path
    def append_new_name_contour(self,number,this_contoursx,this_contoursy,dir):
        #buffer
        self.img_num.append(number)
        self.contoursx.append(this_contoursx)
        self.contoursy.append(this_contoursy)

        #save the data 
        save_path = dir #+ "seg label pkl/"
        with open(save_path+'contours.pkl', 'wb') as f:
            pickle.dump(self , f, pickle.HIGHEST_PROTOCOL)
        pass
    def check(self):
        saved_path  = self.save_contour_dir
        data = pickle.load(open(saved_path+'contours.pkl','rb'),encoding='iso-8859-1')
        file_len = len(data.img_num)
        for num in range(file_len):
                name = data.img_num[num]
                img_path = self.save_img_dir +  str(name) + ".jpg"
                img_or = cv2.imread(img_path)
                img1  =   cv2.cvtColor(img_or, cv2.COLOR_BGR2GRAY)
                H,W = img1.shape
                #just use the first contour 
                contour0x  = data.contoursx[num]
                contour0y  = data.contoursy[num]
                # draw this original contour 
                display = Basic_Operator.draw_coordinates_color(img_or,contour0x,contour0y,1)
                cv2.imshow('origin',display.astype(np.uint8))
                cv2.waitKey(10)   

        pass
    #display the the contour  gray 
    def display_contour(self,img,contourx,contoury,title):
        if self.display_flag ==True:
                display = img
                contour0 = contour
                for j in range (len(contoury)):
                     #path0l[path0x[j]]
                     display[int(contoury[j])+1,contourx[j]]=display[int(contoury[j])-1,contourx[j] ]=display[int(contoury[j]),contourx[j] ]=254
                cv2.imshow(title,display.astype(np.uint8) )
                cv2.waitKey(10)   
        pass
    

    


    # to generate synthetic background with a number of origin img and return a image with size of H W
    def generate_background_image1(self,number,H,W):
        OriginalpathDirlist = os.listdir(self.back_ground_root)    # 
        image_list = [None]*number
        w_list = [None]* number
        h_list = [None]* number

        for i in range(number):
            sample = random.sample(OriginalpathDirlist, 1)  # 
            Sample_path = self.back_ground_root +   sample[0]
            original_IMG = cv2.imread(Sample_path)
            original_IMG  =   cv2.cvtColor(original_IMG, cv2.COLOR_BGR2GRAY)
            h,w  =  original_IMG.shape
            image_list[i] = original_IMG
            w_list[i] = w
            h_list[i] = h

        w_sum = sum(w_list)
        new  = np.zeros((H,W))
        #generate line by line 
        for i in range(W):
            #random select a source
            pic_it = int( np.random.random_sample()*number)
            pic_it = np.clip(pic_it,0,number-1) 
            img=image_list[pic_it]
            #random select a A-line
            line_it = int( np.random.random_sample()*w_list[pic_it])
            line_it = np.clip(line_it,0,w_list[pic_it]-1) 
            source_line = img[:,line_it]
            source_h  = h_list[pic_it]
            new[:,i] = self.fill_lv_with_sv1(source_line,H)
        return new
    
    

        

    def generate(self):
        file_len = len(self.origin_data.img_num)
        #for num  in  self.origin_data.img_num:
        img_id =1
        for n in range(250):
            for num in range(file_len):
                name = self.origin_data.img_num[num]
                img_path = self.database_root + "pic/" + name + ".jpg"
                img_or = cv2.imread(img_path)
                img1  =   cv2.cvtColor(img_or, cv2.COLOR_BGR2GRAY)
                H,W = img1.shape
                #just use the first contour 
                contour0x  = self.origin_data.contoursx[num][0]
                contour0y  = self.origin_data.contoursy[num][0]
                # draw this original contour 
                display = Basic_Operator.draw_coordinates_color(img_or,contour0x,contour0y,1)
                cv2.imshow('origin',display.astype(np.uint8))
            
                #new_contourx=contour0x  +200
                #new_contoury=contour0y-200

                H_new = 1024
                W_new = 1024
                #generate the signal 

                new_contourx,new_contoury = Basic_Operator.random_shape_contour(H_new,W_new,contour0x,contour0y)

                num_points = len(new_contourx)
                #patch_l = generator.generate_background_image1(1,H_new,new_contourx[0])
                patch_l = Basic_Operator .generate_background_image2(img1,contour0x,contour0y,H_new,new_contourx[0])
                patch_r = Basic_Operator .generate_background_image2(img1,contour0x,contour0y,H_new,W_new -new_contourx[num_points-1])

                #patch_r = generator.generate_background_image1(1,H_new,W_new -new_contourx[num_points-1])
                #warp the contour 
                patch = Basic_Operator .generate_patch_with_contour(img1,H_new,contour0x,contour0y,
                                                     new_contourx,new_contoury)
                #patch = Basic_Operator .generate_patch_base_origin(img1,H_new,contour0x,contour0y,
                #                                     new_contourx,new_contoury)
                new_image=np.append(patch_l,patch,axis=1) 
                new_image=np.append(new_image,patch_r,axis=1) 
                #new_image =new_image*0.8
                cv2.imshow('w1',new_image.astype(np.uint8))

                #self.display_contour(new_image,new_contourx,new_contoury,'warped')  
                display  = Basic_Operator.gray2rgb(new_image)
                display  = Basic_Operator.draw_coordinates_color(display,new_contourx,new_contoury,1)

                #display = Basic_Operator.draw_coordinates_color(img_or,contour0x,contour0y,1)
                cv2.imshow('color',display.astype(np.uint8))
                cv2.waitKey(10)   

                print(str(name))
                self.append_new_name_contour(img_id,new_contourx,new_contoury,self.save_contour_dir)
                cv2.imwrite(self.save_img_dir  + str(img_id) +".jpg",new_image )
                img_id +=1

            


                pass


if __name__ == '__main__':
    generator  = Generator_Contour()
    generator.generate()

    generator.check()

    #back = generator.generate_background_image1(3,1024,1024)
    #cv2.imshow('origin',back.astype(np.uint8))
    cv2.waitKey(10) 
