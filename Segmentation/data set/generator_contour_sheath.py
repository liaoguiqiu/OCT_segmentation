#7th October 2020
#update the gnerator to add the situation with sheath 
import cv2
import numpy as np
import os
import random
from matplotlib.pyplot import *
#from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import matplotlib.pyplot as plt
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
        self.cv_display = False
        self.origin_data = Save_Contour_pkl()
        #self.database_root = "../../OCT/beam_scanning/Data Set Reorganize/VARY/"
        self.image_dir ="D:/Deep learning/dataset/label data/img/"
        self.pkl_dir ="D:/Deep learning/dataset/label data/seg label pkl/"
        self.save_image_dir ="D:/Deep learning/dataset/label data/img_generate/" # this dir just save all together
        self.save_image_dir_devi ="D:/Deep learning/dataset/label data/img_genetate_devi/" # this dir devide the generated images

        self.save_pkl_dir ="D:/Deep learning/dataset/label data/pkl_generate/"
        #self.origin_data =self.origin_data.read_data(self.pkl_dir)
        #self.origin_data = []

        self.back_ground_root  =  "../../"     + "saved_background_for_generator/"

        #self.save_img_dir = "../../"     + "saved_generated_contour/"
        #self.save_contour_dir = "../../"     + "saved_stastics_coutour_generated/"
        self.display_flag =True
        self.img_num= []
        self.contoursx = []
        self.contoursy = []

        #get all the folder
        self.all_dir_list = os.listdir(self.pkl_dir)
        self.folder_num = len(self.all_dir_list)
        # create the buffer list
        self.folder_list = [None]*self.folder_num
        self.signal = [None]*self.folder_num

        # create a detail foldeer list to save the generated images

        for subfold in self.all_dir_list:
            save_sub =  self. save_image_dir_devi + subfold+'/'
            if not os.path.exists(save_sub):
                os.makedirs(save_sub)
        # create all  the folder list and their data list

        #number_i = 0
        ## all_dir_list is subfolder list 
        ##creat the image list point to the STASTICS TIS  list
        ##saved_stastics = Generator_Contour()
        ##read all the folder list
        #for subfold in self.all_dir_list:
        #    #if(number_i==0):
        #    this_folder_list =  os.listdir(os.path.join(self.pkl_dir, subfold))
        #    this_folder_list2 = [ self.pkl_dir +subfold + "/" + pointer for pointer in this_folder_list]
        #    self.folder_list[number_i] = this_folder_list2

        #    #change the dir firstly before read
        #    #saved_stastics.all_statics_dir = os.path.join(self.signalroot, subfold, 'contour.pkl')
        #    this_contour_dir =  self.pkl_dir+ subfold+'/'+ 'contours.pkl' # for both linux and window

        #    self.signal[number_i]  =  self.read_data(this_contour_dir)
        #    number_i +=1
            #read the folder list finished  get the folder list and all saved path
    
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
        saved_path  = self.save_pkl_dir
        data = pickle.load(open(saved_path+'contours.pkl','rb'),encoding='iso-8859-1')
        file_len = len(data.img_num)
        for num in range(file_len):
                name = data.img_num[num]
                img_path = self.save_image_dir +  str(name) + ".jpg"
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
    # to calculate the statitic distrubution of the dast 
    def stastics (self):
        #for num  in  self.origin_data.img_num:
        img_id =1
        distance_ori = []
        distance_new = [] 
        contact_r_ori =[] # the ratio of the contact 
        contact_r_new =[] # the ratio of the contact 
 
            #number_i = 0          
        for subfold in self.all_dir_list:
         
            #saved_stastics.all_statics_dir = os.path.join(self.signalroot, subfold, 'contour.pkl')
            this_contour_dir =  self.pkl_dir+ subfold+'/'  # for both linux and window

            self.origin_data =self.origin_data.read_data(this_contour_dir)  # this original data
            #number_i +=1
            file_len = len(self.origin_data.img_num)

            for num in range(file_len):
                name = self.origin_data.img_num[num]
                img_path = self.image_dir+ subfold+'/' + name + ".jpg"
                img_or = cv2.imread(img_path)
                img1  =   cv2.cvtColor(img_or, cv2.COLOR_BGR2GRAY)
                H,W = img1.shape
                #just use the first contour 
                #contour0x  = self.origin_data.contoursx[num][0]
                #contour0y  = self.origin_data.contoursy[num][0]
                contourx  = self.origin_data.contoursx[num]
                contoury  = self.origin_data.contoursy[num]


                # fill in the background area with H value when contour is not fully labeld , 
                contourx[0],contoury[0] = Basic_Operator2.re_fresh_path(contourx[0],contoury[0],H,W)
                contourx[1],contoury[1] = Basic_Operator2.re_fresh_path(contourx[1],contoury[1],H,W)

                # uniform the initial path 
                distance_ori.append((contoury[1]-contoury[0])/H)
                # 


                if self.cv_display ==True:
                    # draw this original contour 
                    display = Basic_Operator.draw_coordinates_color(img_or,contourx[0],contoury[0],1) # draw the sheath
                    display = Basic_Operator.draw_coordinates_color(img_or,contourx[1],contoury[1],2) # draw the tissue
                    cv2.imshow('origin',display.astype(np.uint8))
            
                #new_contourx=contour0x +200
                #new_contoury=contour0y-200

                H_new = H
                W_new = W
                # genrate the new sheath contour
                sheath_x,sheath_y = Basic_Operator2.random_sheath_contour(H_new,W_new,contourx[0],contoury[0])

             
                 

                #generate the signal 
                dc1 =np.random.random_sample()*100
                dc1  = int(dc1)%2
                if dc1!=0: 
                    new_contourx,new_contoury = Basic_Operator2.random_shape_contour(H,W,H_new,W_new,sheath_x,sheath_y,contourx[1],contoury[1])
                else:
                    new_contourx,new_contoury = Basic_Operator2.random_shape_contour2(H,W,H_new,W_new,sheath_x,sheath_y,contourx[1],contoury[1])

                # fill in the blank area 

                # aligh and cut the signal
                #left= max(sheath_x[0],new_contourx[0])
                #right=min(sheath_x[len(sheath_x)-1],  new_contourx[len(new_contourx)-1])
                # fill in the background area with H value when contour is not fully labeld , 
                sheath_x,sheath_y = Basic_Operator2.re_fresh_path(sheath_x,sheath_y,H_new,W_new)
                new_contourx , new_contoury = Basic_Operator2.re_fresh_path(new_contourx,new_contoury,H_new ,W_new  )
                distance_new .append((new_contoury -sheath_y)/H_new)
                #min_b  = int(np.max(contoury[0]))
                #max_b  = int(np.min(contoury[1]))
       
                #new_cx  = [None]*2
                #new_cy   = [None]*2
                #new_cx[0]  = sheath_x
                #new_cy[0]  = sheath_y
                #new_cx[1]  = new_contourx
                #new_cy[1]  = new_contoury

                print(str(name))
                #self.append_new_name_contour(img_id,new_cx,new_cy,self.save_pkl_dir)
                # save them altogether 
                # save them separetly 
                #if not os.path.exists(directory):
                # os.makedirs(directory)

                img_id +=1

                pass

            #time.sleep(1)

            #sns.set_style('darkgrid')
            #dis = np.concatenate(distance_ori).flat
            #plt.figure()

            #sns.distplot(dis)
            #time.sleep(0.1)
        #d =  np.random.sample(1000)*10
        time.sleep(0.1)

        sns.set_style('darkgrid')
        dis = np.concatenate(distance_ori).flat
        plt.figure()
        sns.distplot(dis)
        time.sleep(0.1)

        time.sleep(0.1)

        sns.set_style('darkgrid')
        plt.figure()

        dis = np.concatenate(distance_new ).flat
        clear_list =   [x for x in dis if (x>0 and x <= 0.75)]

        sns.distplot(clear_list)

        plt.figure()
        sns.distplot(dis)
        time.sleep(0.1)



    


  

        

    def generate(self):
        #for num  in  self.origin_data.img_num:
        img_id =1
        # all_dir_list is subfolder list 
        #creat the image list point to the STASTICS TIS  list
        #saved_stastics = Generator_Contour()
        #read all the folder list
            #number_i = 0          
        for subfold in self.all_dir_list:
            #if(number_i==0):
            #this_folder_list =  os.listdir(os.path.join(self.pkl_dir, subfold))
            #this_folder_list2 = [ self.pkl_dir +subfold + "/" + pointer for pointer in this_folder_list]
            #self.folder_list[number_i] = this_folder_list2

            #change the dir firstly before read
            #saved_stastics.all_statics_dir = os.path.join(self.signalroot, subfold, 'contour.pkl')
            this_contour_dir =  self.pkl_dir+ subfold+'/'  # for both linux and window

            self.origin_data =self.origin_data.read_data(this_contour_dir)  # this original data
            #number_i +=1
            file_len = len(self.origin_data.img_num)

            repeat = int(100/file_len) # repeat to balance
            if repeat<1:

                repeat = 1

            img_id_devi = 1  # this image id is for one categrot/ subfolder 

            for n in range(repeat):  # repeat this to all data

                for num in range(file_len):
                    name = self.origin_data.img_num[num]
                    img_path = self.image_dir+ subfold+'/' + name + ".jpg"
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
                    if self.cv_display ==True:
                        cv2.imshow('origin',display.astype(np.uint8))
            
                    #new_contourx=contour0x  +200
                    #new_contoury=contour0y-200

                    H_new = H
                    W_new = W
                    # genrate the new sheath contour
                    sheath_x,sheath_y = Basic_Operator2.random_sheath_contour(H_new,W_new,contourx[0],contoury[0])

                    New_img , mask = Basic_Operator2. fill_sheath_with_contour(img1,H_new,W_new,contourx[0],contoury[0],
                                        sheath_x,sheath_y)
                    if self.cv_display ==True:

                        cv2.imshow('shealth',New_img.astype(np.uint8))

                    #generate the signal 
                    new_contourx,new_contoury = Basic_Operator2.random_shape_contour3(H,W,H_new,W_new,sheath_x,sheath_y,contourx[1],contoury[1])
                    New_img , mask  = Basic_Operator2. fill_patch_base_origin2(img1,H_new,contourx[1],contoury[1],
                                        new_contourx,new_contoury,New_img , mask )
                
                
                    if self.cv_display ==True:
 
                        cv2.imshow('mask',New_img.astype(np.uint8))

                    #----------fill in the blank area today 
                    #----------fill in the blank area today 
                    #----------fill in the blank area today 
                    # fill in the blank area

                    contourx[0],contoury[0] = Basic_Operator2.re_fresh_path(contourx[0],contoury[0],H,W)

                    contourx[1],contoury[1] = Basic_Operator2.re_fresh_path(contourx[1],contoury[1],H,W)
                    min_b  = int(np.max(contoury[0]))
                    max_b  = int(np.min(contoury[1]))
                    backimage  =  Basic_Operator2.pure_background(img1 ,contourx,contoury, H_new,W_new)
                    if self.cv_display ==True:

                        cv2.imshow('back',backimage.astype(np.uint8))
                    combin = Basic_Operator2.add_original_back(New_img,backimage,mask)
                    #combin= Basic_Operator.add_speckle_or_not(combin)
                    #combin= Basic_Operator.add_noise_or_not(combin)
                    #combin = Basic_Operator.add_gap_or_not(combin)

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
                    # save them altogether 
                    cv2.imwrite(self.save_image_dir  + str(img_id) +".jpg",combin )
                    cv2.imwrite(self.save_image_dir_devi + subfold + '/' + str(img_id_devi) +".jpg",combin ) # save them separately


                    # save them separetly 
                    #if not os.path.exists(directory):
                    # os.makedirs(directory)

                    img_id +=1
                    img_id_devi +=1 
            


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

        imgbase_dir = "../../../../../"  + "Deep learning/dataset/For_contour_sheath_train/train_OLG/pic/"
        labelbase_dir = "../../../../../"  + "Deep learning/dataset/For_contour_sheath_train/train_OLG/label/"

        while(1):
            generator  = Generator_Contour_sheath()

            talker=talker.read_data(com_dir)
            # only use for the first ytime
            #talker.training=1  # only use for the first ytime
            #talker.writing=2  # only use for the first ytime
            #talker.pending = 0 # only use for the first ytime
            if talker.training==1 and talker.writing==2: # check if 2 need writing
                if talker.pending == 0 :
                    generator.save_image_dir = imgbase_dir+"2/"
                    generator.save_pkl_dir =  labelbase_dir+"2/"

                    generator.generate() # generate

                    talker.writing=1
                    talker.pending=1
                    talker.save_data(com_dir)
            # set break point here for first time
            if talker.training==2 and talker.writing==1: # check if 2 need writing
                if talker.pending == 0 :
                    generator.save_image_dir = imgbase_dir+"1/"
                    generator.save_pkl_dir =  labelbase_dir+"1/"

                    generator.generate() # generate

                    talker.writing=2
                    talker.pending=1
                    talker.save_data(com_dir)
            cv2.waitKey(1000)   
            print("waiting")
    generator.stastics()
    generator.generate()
    
    generator.check()

    #back = generator.generate_background_image1(3,1024,1024)
    #cv2.imshow('origin',back.astype(np.uint8))
    cv2.waitKey(10) 
