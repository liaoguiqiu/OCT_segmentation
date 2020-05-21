import cv2
import math
import numpy as np
import os
import random
from matplotlib.pyplot import *
#from mpl_toolkits.mplot3d import Axes3D
from median_filter_special import  myfilter
from cost_matrix import  COSTMtrix

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

class Save_Contour_pkl(object):

    def __init__(self ):

        self.database_root = "../../OCT/beam_scanning/Data Set Reorganize/NORMAL/"
        #check or create this path
        #self.self_check_path_create(self.signal_data_path)
        self.img_num= []
        self.contours = []

    def self_check_path_create(self,directory):
        try:
            os.stat(directory)
        except:
            os.mkdir(directory)  
   # add new step of all signals
    def append_new_name_contour(self,number,this_contours,dir):
        #buffer
        self.img_num.append(number)
        self.contours.append(this_contours)
        #save the data 
        save_path = dir + "seg label pkl/"
        with open(save_path+'contours.pkl', 'wb') as f:
            pickle.dump(self , f, pickle.HIGHEST_PROTOCOL)
        pass
    #read from file
    def read_data(self):
        saved_path  = self.database_root  + "seg label pkl/"
        self = pickle.load(open(saved_path+'contours.pkl','rb'),encoding='iso-8859-1')
        return self

class Generator_Contour(object):
    def __init__(self ):
        self.origin_data = Save_Contour_pkl()
        self.database_root = self.origin_data.database_root
        self.origin_data =self.origin_data.read_data()

        self.save_img_dir = "../../"     + "saved_generated_contour/"
        self.save_contour_dir = "../../"     + "saved_stastics_coutour_generated/"
        self.display_flag =True
        #check or create this path
    def display_contour(self,img,contour,title):
        if self.display_flag ==True:
                display = img
                contour0 = contour
                for j in range (len(contour)):
                             #path0l[path0x[j]]
                     display[int(contour0[j])+1,j]=display[int(contour0[j])-1,j ]=display[int(contour0[j]),j ]=0
                cv2.imshow(title,display.astype(np.uint8) )
                cv2.waitKey(10)   
        pass
    def warp_padding(self,img,contour0,new_contour):
        shift_vector =  new_contour  - contour0
        new_image  = img
        for iter in range(len(shift_vector)):
            lineshift= int(shift_vector[iter] )
            new_image[:,iter] =np.roll( img[:,iter] ,lineshift)
            # The carachter within the contour need special stratigies to maintain 
            if(lineshift>0):#
                origin_point  = int(contour0[iter])
                new_image[0:lineshift,iter]= signal.resample(img[0:origin_point,iter], 
                                                             lineshift)
                pass
        return new_image
    def generate(self):
        file_len = len(self.origin_data.img_num)
        #for num  in  self.origin_data.img_num:
        for num in range(file_len):
            name = self.origin_data.img_num[num]
            img_path = self.database_root + "pic/" + name + ".jpg"
            img1 = cv2.imread(img_path)
            img1  =   cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            H,W = img1.shape
            contour0  = self.origin_data.contours[num][0]
            # draw this original contour 
            self.display_contour(img1,contour0,'origin')  
            print(str(name))
            sample = np.arange(W)
            new_contour = np.sin( np.pi/W * sample)
            new_contour =500 - 200*new_contour
            #warp the contour 
            new_image = self.warp_padding(img1,contour0,new_contour)
            new_image =new_image*0.5
            self.display_contour(new_image,new_contour,'warped')  
            


            pass


if __name__ == '__main__':
    generator  = Generator_Contour()
    generator.generate()
