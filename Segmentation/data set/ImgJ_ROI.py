from read_roi import read_roi_file
from read_roi import read_roi_zip

# for vs proj the path jump depends on the proj file position
read_file_dir = "../../saved_ROI_Imagj/0539-0183.roi"
read_zip_dir  = "../../saved_ROI_Imagj/319-1 ROI.zip"

roi = read_roi_file(read_file_dir)
# or
rois = read_roi_zip(read_zip_dir)
roi = read_roi_file(read_file_dir)
