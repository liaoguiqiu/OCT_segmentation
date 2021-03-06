import cv2
import numpy as np
import os
from analy import MY_ANALYSIS
from analy import Save_signal_enum
from scipy import signal 
import random
from random import seed
from median_filter_special import myfilter
from Correct_sequence_iteration import VIDEO_PEOCESS
from  path_finding import PATH
from scipy.ndimage import gaussian_filter1d

from cost_matrix import COSTMtrix ,Overall_shiftting_WinLen , Window_LEN
class DATA_Generator(object):
     def __init__(self):
        self.original_root = "../../saved_original_for_generator/"
        self.data_pair1_root = "../../saved_pair1/"
        self.data_pair2_root = "../../saved_pair2/"
        self.data_mat_root = "../../saved_matrix/"
        self.data_mat_root_origin = "../../saved_matrix_unprocessed/"

        self.data_signal_root  = "../../saved_stastics_for_generator/"
        self.H  = 1024
        self.W = 780
        # read the signals  just use the existing path
        self.saved_stastics = MY_ANALYSIS()
        self.saved_stastics.all_statics_dir = os.path.join(self.data_signal_root, 'signals.pkl')

        self.path_DS =  self.saved_stastics.read_my_signal_results()
        self.path_DS.all_statics_dir  =  self.saved_stastics.all_statics_dir


        #the  validation functionfor check the matrix and can also be used for validate the correction result
     def validation(self,original_IMG,Shifted_IMG,path,Image_ID):
        #Costmatrix,shift_used = COSTMtrix.matrix_cal_corre_full_version3_2GPU(original_IMG,Shifted_IMG,0) 
        Costmatrix,shift_used = COSTMtrix.matrix_cal_corre_block_version3_3GPU(original_IMG,Shifted_IMG,0) 

        # Costmatrix  = myfilter.gauss_filter_s(Costmatrix) # smooth matrix
        #tradition way to find path
 
        start_point= PATH.find_the_starting(Costmatrix) # starting point for path searching

        #path_tradition,pathcost1  = PATH.search_a_path(Costmatrix,start_point) # get the path and average cost of the path
        #path_deep,path_cost2=PATH.search_a_path_Deep_Mat2longpath(Costmatrix) # get the path and average cost of the path
        path_deep,path_cost2=PATH.search_a_path_deep_multiscal_small_window(Costmatrix) # get the path and average cost of the path
        
        path_deep = gaussian_filter1d(path_deep,3) # smooth the path 

        ##middle_point  =  PATH.calculate_ave_mid(mat)
        #path1,path_cost1=PATH.search_a_path(mat,start_point) # get the path and average cost of the path
        show1 =  Costmatrix 
        cv2.imwrite(self.data_mat_root_origin  + str(Image_ID) +".jpg", show1)

        for i in range ( len(path)):
            painter = min(path[i],Window_LEN-1)
            #painter2= min(path_tradition[i],Window_LEN-1)
            painter3 = min(path_deep[i],Window_LEN-1) 
            show1[int(painter),i]=128
            #show1[int(painter2),i]=128
            show1[int(painter3),i]=254

        cv2.imwrite( self.data_mat_root  + str(Image_ID) +".jpg", show1)


         

     def generate_NURD(self):
         #read one from the original
            #random select one IMG frome the oringinal 
        read_id = 0
        Len_steam =5
        steam=np.zeros((Len_steam,self.H,self.W)) # create video buffer
        while (1):
            OriginalpathDirlist = os.listdir(self.original_root)    # 
            sample = random.sample(OriginalpathDirlist, 1)  # 
            Sample_path = self.original_root +   sample[0]
            original_IMG = cv2.imread(Sample_path)
            original_IMG  =   cv2.cvtColor(original_IMG, cv2.COLOR_BGR2GRAY)
            original_IMG = cv2.resize(original_IMG, (self.W,self.H), interpolation=cv2.INTER_AREA)

            #read the path and Image number from the signal file
            #get the Id of image which should be poibnt to
            Image_ID = int( self.path_DS.signals[Save_signal_enum.image_iD.value, read_id])
            #get the path
            path  = self.path_DS.path_saving[read_id,:]
            path =  signal.resample(path, self.W)#resample the path
            # create the shifted image
            Shifted_IMG   = VIDEO_PEOCESS.de_distortion(original_IMG,path,Image_ID,0)
            # save all the result
            cv2.imwrite(self.data_pair1_root  + str(Image_ID) +".jpg", original_IMG)
            cv2.imwrite(self.data_pair2_root  + str(Image_ID) +".jpg", Shifted_IMG)
            ## validation 
            self.validation(original_IMG,Shifted_IMG,path,Image_ID) 

            #steam[Len_steam-1,:,:]  = original_IMG  # un-correct 
            #steam[Len_steam-2,:,:]  = Shifted_IMG  # correct 
            #Costmatrix,shift_used = COSTMtrix.matrix_cal_corre_full_version3_2GPU(original_IMG,Shifted_IMG,0) 
            #Costmatrix  = myfilter.gauss_filter_s (Costmatrix) # smooth matrix
            #show1 =  Costmatrix 
            #for i in range ( len(path)):
            #    show1[int(path[i]),i]=254
            #cv2.imwrite(self.data_mat_root  + str(Image_ID) +".jpg", show1)



            print ("[%s]   is processed. test point time is [%f] " % (read_id ,0.1))

            read_id +=1
     def generate_overall_shifting(self):
         #read one from the original
            #random select one IMG frome the oringinal 
        read_id = 0
        Len_steam =5
        #steam=np.zeros((Len_steam,self.H,self.W)) # create video buffer
        while (1):
            random_shifting = random.random() * Overall_shiftting_WinLen
            OriginalpathDirlist = os.listdir(self.original_root)    # 
            sample = random.sample(OriginalpathDirlist, 1)  # 
            Sample_path = self.original_root +   sample[0]
            original_IMG = cv2.imread(Sample_path)
            original_IMG  =   cv2.cvtColor(original_IMG, cv2.COLOR_BGR2GRAY)
            #original_IMG = cv2.resize(original_IMG, (self.W,self.H), interpolation=cv2.INTER_AREA)
            H,W = original_IMG.shape

            #read the path and Image number from the signal file
            #get the Id of image which should be poibnt to
            Image_ID = int( self.path_DS.signals[Save_signal_enum.image_iD.value, read_id])
            #get the path
            path  = self.path_DS.path_saving[read_id,:]
            #change the signal too
            self.path_DS.path_saving[read_id,:] = path* 0 + random_shifting
            path =  signal.resample(path, W)*0 + random_shifting  #resample the path
            
            #resave the signal

            # create the shifted image
            Shifted_IMG   = VIDEO_PEOCESS.de_distortion(original_IMG,path,Image_ID,0)
            # save all the result
            cv2.imwrite(self.data_pair1_root  + str(Image_ID) +".jpg", original_IMG)
            cv2.imwrite(self.data_pair2_root  + str(Image_ID) +".jpg", Shifted_IMG)
            self.path_DS.save()
            self.validation(original_IMG,Shifted_IMG,path,Image_ID) 

            ## validation 
            #steam[Len_steam-1,:,:]  = original_IMG  # un-correct 
            #steam[Len_steam-2,:,:]  = Shifted_IMG  # correct 
            #Costmatrix,shift_used = COSTMtrix.matrix_cal_corre_full_version3_2GPU(original_IMG,Shifted_IMG,0) 
            ##Costmatrix  = myfilter.gauss_filter_s (Costmatrix) # smooth matrix
            #show1 =  Costmatrix 
            #for i in range ( len(path)):
            #    show1[int(path[i]),i]=254
            #cv2.imwrite(self.data_mat_root  + str(Image_ID) +".jpg", show1)



            print ("[%s]   is processed. test point time is [%f] " % (read_id ,0.1))

            read_id +=1

     def generate_NURD_overall_shifting(self):
         pass



if __name__ == '__main__':
        generator   = DATA_Generator()
        generator.generate_NURD ()