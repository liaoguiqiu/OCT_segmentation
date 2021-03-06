from read_roi import read_roi_file
from read_roi import read_roi_zip
import cv2
import math
import numpy as np
import os
import random 
from zipfile import ZipFile
import scipy.signal as signal
import pandas as pd
from generator_contour import Save_Contour_pkl
class  Read_read_check_ROI_label(object):
    def __init__(self ):
        #self.image_dir   = "../../OCT/beam_scanning/Data set/pic/NORMAL-BACKSIDE-center/"
        #self.roi_dir =  "../../OCT/beam_scanning/Data set/seg label/NORMAL-BACKSIDE-center/"
        self.database_root = "../../OCT/beam_scanning/Data Set Reorganize/NORMAL/"
        self.image_dir   = self.database_root + "pic/"
        self.roi_dir =  self.database_root + "seg label/"
        self.img_num = 0
        self.contours = []
        self.saver = Save_Contour_pkl()

    def check_one_folder (self):
        for i in os.listdir(self.roi_dir):
    #for i in os.listdir("E:\\estimagine\\vs_project\\PythonApplication_data_au\\pic\\"):
        # 分离文件名与后缀
            a, b = os.path.splitext(i)
            # 如果后缀名是“.xml”就旋转related的图像
            if b == ".zip"or b== ".ZIP":
                img_path = self.image_dir + a + ".jpg"
                img1 = cv2.imread(img_path)
                if img1 is None:
                    print ("no_img for this zip")
                else:
                    roi_dir = self.roi_dir + a  +b
                    with ZipFile(roi_dir, 'r') as zipObj:
                           # Get list of files names in zip
                           #listOfiles = zipObj.namelist()
                           # this line of code is importanct sice the the formmer one will change the sequence 
                           listOfiles = zipObj.infolist()

                           line_name0,_ = os.path.splitext(listOfiles[0].filename)
                           number0   = int(line_name0.split("-")[0])
                 
                           line_name1,_ = os.path.splitext(listOfiles[1].filename)
                           number1   = int(line_name1.split("-")[0])

                           line_name2,_ = os.path.splitext(listOfiles[2].filename)
                           number2   = int(line_name2.split("-")[0])

                           line_name3,_ = os.path.splitext(listOfiles[3].filename)
                           number3   = int(line_name3.split("-")[0])

                           number_list= [number0,number1,number2,number3]
                           new_index  = np.argsort( number_list)

                           name_list = [line_name0,line_name1,line_name2,line_name3]
                           line_name0 = name_list[new_index[0]]
                           line_name1 = name_list[new_index[1]]
                           line_name2 = name_list[new_index[2]]
                           line_name3 = name_list[new_index[3]]




                        
                    rois = read_roi_zip(roi_dir)
                     
                    gray  =   cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
                    H,W   = gray.shape
                    path0y = rois[line_name0]['y']
                    path0x = rois[line_name0]['x']
                    path0l =  np.ones(W) * np.nan
                    path1y = rois[line_name1]['y']
                    path1x = rois[line_name1]['x']
                    path1l =  np.ones(W) * np.nan
                    path2y = rois[line_name2]['y']
                    path2x = rois[line_name2]['x']
                    path2l =  np.ones(W) * np.nan
                    path3y = rois[line_name3]['y']
                    path3x = rois[line_name3]['x']
                    path3l =  np.ones(W) * np.nan
                    for j in range (len(path0x)):
                         this_index = np.clip(  path0x[j]-1, 0,W-1)
                         path0l[this_index] = float (path0y[j] -1)
                    for j in range (len(path1x)):
                         this_index = np.clip(  path1x[j]-1, 0,W-1)
                         path1l[this_index] = float (path1y[j] -1)
                    for j in range (len(path2x)):
                         this_index = np.clip(  path2x[j]-1, 0,W-1)
                         path2l[this_index] = float (path2y[j] -1)                 
                    for j in range (len(path3x)):
                         this_index = np.clip(  path3x[j]-1, 0,W-1)
                         path3l[this_index] = float (path3y[j] -1)

                    add_3   = np.append(path0l[::-1],path0l,axis=0) # cascade
                    add_3   = np.append(add_3,path0l[::-1],axis=0) # cascade
                    s = pd.Series(add_3)
                    path0ln = s.interpolate(  )
                    path0ln = path0ln[W:2*W].to_numpy() 

                    add_3   = np.append(path1l[::-1],path1l,axis=0) # cascade
                    add_3   = np.append(add_3,path1l[::-1],axis=0) # cascade
                    s = pd.Series(add_3)
                    path1ln = s.interpolate( )
                    path1ln = path1ln[W:2*W].to_numpy() 

                    add_3   = np.append(path2l[::-1],path2l,axis=0) # cascade
                    add_3   = np.append(add_3,path2l[::-1],axis=0) # cascade
                    s = pd.Series(add_3)
                    path2ln = s.interpolate(  )
                    path2ln = path2ln[W:2*W].to_numpy() 

                    add_3   = np.append(path3l[::-1],path3l,axis=0) # cascade
                    add_3   = np.append(add_3,path3l[::-1],axis=0) # cascade
                    s = pd.Series(add_3)
                    path3ln = s.interpolate(  )
                    path3ln = path3ln[W:2*W].to_numpy() 

                    #path0  = signal.resample(path0, W)
                    for j in range (len(path0ln)):
                         #path0l[path0x[j]]
                         img1[int(path0ln[j])+1,j,:]=img1[int(path0ln[j])-1,j,:]=img1[int(path0ln[j]),j,:]=[254,0,0]
                         img1[int(path1ln[j]+1),j,:]=img1[int(path1ln[j]-1),j,:]=img1[int(path1ln[j]),j,:]=[0,254,0]
                         img1[int(path2ln[j]+1),j,:]=img1[int(path2ln[j]-1),j,:]=img1[int(path2ln[j]),j,:]=[0,0,254]
                         img1[int(path3ln[j]+1),j,:]=img1[int(path3ln[j]-1),j,:]=img1[int(path3ln[j]),j,:]=[254,254,254]
                    
                    #save this result 
                    self.img_num = a
                    self.contours = [path0ln, path1ln, path2ln, path3ln]
                    self.saver.append_new_name_contour (self.img_num,self.contours,self.database_root)

                    cv2.imshow('pic',img1)
                    print(str(a))
                    cv2.waitKey(10) 
                    
 
        #return super().__init__(*args, **kwargs)
# for vs proj the path jump depends on the proj file position

if __name__ == '__main__':
    cheker  = Read_read_check_ROI_label()
    cheker.check_one_folder() 
 

    #roi = read_roi_file(read_file_dir)
