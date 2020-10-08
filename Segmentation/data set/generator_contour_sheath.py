#7th October 2020
#update the gnerator to add the situation with sheath 
import cv2
import numpy as np
import os
import random
from matplotlib.pyplot import *
#from mpl_toolkits.mplot3d import Axes3D

#PythonETpackage for xml file edition
import pickle
from operater import Basic_Operator
from operator2 import Basic_Operator2

# this is used  to communicate with trainner py
class Communicate(object):
    def __init__(self ):
        #set = Read_read_check_ROI_label()
        #self.database_root = set.database_root
        #check or create this path
        #self.self_check_path_create(self.signal_data_path)
        self.training= 1
        self.writing = 2
        self.pending = 1
    def change_state(self):
        if self.writing ==1:
           self.writing =0
        pass
    def read_data(self,dir):
        saved_path  = dir  + 'protocol.pkl'
        self = pickle.load(open(saved_path,'rb'),encoding='iso-8859-1')
        return self
    def save_data(self,dir):
        #save the data 
        save_path = dir + 'protocol.pkl'
        with open(save_path , 'wb') as f:
            pickle.dump(self , f, pickle.HIGHEST_PROTOCOL)
        pass


#from ImgJ_ROI2 import Read_read_check_ROI_label
#for this function this is to save the laeyrers
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
        saved_path  = base_root   
        self = pickle.load(open(saved_path+'contours.pkl','rb'),encoding='iso-8859-1')
        return self

class Generator_Contour_sheath(object):
    def __init__(self ):
        self.OLG_flag =False
        self.origin_data = Save_Contour_pkl()
        #self.database_root = "../../OCT/beam_scanning/Data Set Reorganize/VARY/"
        self.image_dir ="D:/Deep learning/dataset/label data/img/"
        self.pkl_dir ="D:/Deep learning/dataset/label data/seg label pkl/"
        self.save_image_dir ="D:/Deep learning/dataset/label data/img_generate/"
        self.save_pkl_dir ="D:/Deep learning/dataset/label data/pkl_generate/"
        self.origin_data =self.origin_data.read_data(self.pkl_dir)
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
        for n in range(50):
            for num in range(file_len):
                name = self.origin_data.img_num[num]
                img_path = self.image_dir+ name + ".jpg"
                img_or = cv2.imread(img_path)
                img1  =   cv2.cvtColor(img_or, cv2.COLOR_BGR2GRAY)
                H,W = img1.shape
                #just use the first contour 
                #contour0x  = self.origin_data.contoursx[num][0]
                #contour0y  = self.origin_data.contoursy[num][0]
                contourx  = self.origin_data.contoursx[num]
                contoury  = self.origin_data.contoursy[num]
                # draw this original contour 
                display = Basic_Operator.draw_coordinates_color(img_or,contourx[0],contoury[0],1) # draw the sheath
                display = Basic_Operator.draw_coordinates_color(img_or,contourx[1],contoury[1],2) # draw the tissue

                cv2.imshow('origin',display.astype(np.uint8))
            
                #new_contourx=contour0x  +200
                #new_contoury=contour0y-200

                H_new = H
                W_new = W
                # genrate the new sheath contour
                sheath_x,sheath_y = Basic_Operator2.random_sheath_contour(H_new,W_new,contourx[0],contoury[0])

                New_img , mask = Basic_Operator2. fill_sheath_with_contour(img1,H_new,W_new,contourx[0],contoury[0],
                                    sheath_x,sheath_y)
                cv2.imshow('shealth',New_img.astype(np.uint8))

                #generate the signal 
                new_contourx,new_contoury = Basic_Operator2.random_shape_contour(H,W,H_new,W_new,sheath_x,sheath_y,contourx[1],contoury[1])
                New_img , mask  = Basic_Operator2. fill_patch_base_origin(img1,H_new,contourx[1],contoury[1],
                                    new_contourx,new_contoury,New_img , mask )
                
                
 
                cv2.imshow('mask',New_img.astype(np.uint8))

                #----------fill in the blank area today 
                #----------fill in the blank area today 
                #----------fill in the blank area today 
                min_b  = int(np.max(contoury[0]))
                max_b  = int(np.min(contoury[1]))
                backimage  =  Basic_Operator2.pure_background(img1 ,contourx,contoury, H_new,W_new)
                cv2.imshow('back',backimage.astype(np.uint8))
                combin = Basic_Operator2.add_original_back(New_img,backimage,mask)
                RGB_imag = Basic_Operator.gray2rgb(combin) 
                display = Basic_Operator.draw_coordinates_color(RGB_imag,new_contourx,new_contoury,2) # draw the tissue
                display = Basic_Operator.draw_coordinates_color(RGB_imag,sheath_x,sheath_y,1) # draw the tissue

                cv2.imshow('all',display.astype(np.uint8))
                 
                cv2.waitKey(10)   
                new_cx  = [None]*2
                new_cy   = [None]*2
                new_cx[0]  = sheath_x
                new_cy[0]  = sheath_y
                new_cx[1]  = new_contourx
                new_cy[1]  = new_contoury

                print(str(name))
                self.append_new_name_contour(img_id,new_cx,new_cy,self.save_pkl_dir)
                cv2.imwrite(self.save_image_dir  + str(img_id) +".jpg",combin )
                img_id +=1

            


                pass


if __name__ == '__main__':
    generator  = Generator_Contour_sheath()
    if generator.OLG_flag ==True:
        talker = Communicate()
        com_dir = "../../../../../" + "Deep learning/dataset/telecom/"

        talker=talker.read_data(com_dir)
        #initialize the protocol
        #talker.pending = 1
        #talker=talker.save_data(com_dir)

        #generator.save_img_dir = "../../../../../"  + "Deep learning/dataset/"
        #generator.save_contour_dir = "../../"     + "saved_stastics_coutour_generated/"

        imgbase_dir = "../../../../../"  + "Deep learning/dataset/For_contour_train/pic/"
        labelbase_dir = "../../../../../"  + "Deep learning/dataset/For_contour_train/label/"

        while(1):
            generator  = Generator_Contour()

            talker=talker.read_data(com_dir)

            if talker.training==1 and talker.writing==2: # check if 2 need writing
                if talker.pending == 0 :
                    generator.save_img_dir = imgbase_dir+"2/"
                    generator.save_contour_dir =  labelbase_dir+"2/"

                    generator.generate() # generate

                    talker.writing=1
                    talker.pending=1
                    talker.save_data(com_dir)
            if talker.training==2 and talker.writing==1: # check if 2 need writing
                if talker.pending == 0 :
                    generator.save_img_dir = imgbase_dir+"1/"
                    generator.save_contour_dir =  labelbase_dir+"1/"

                    generator.generate() # generate

                    talker.writing=2
                    talker.pending=1
                    talker.save_data(com_dir)
            cv2.waitKey(1000)   
            print("waiting")



    
    generator.generate()
    
    generator.check()

    #back = generator.generate_background_image1(3,1024,1024)
    #cv2.imshow('origin',back.astype(np.uint8))
    cv2.waitKey(10) 
