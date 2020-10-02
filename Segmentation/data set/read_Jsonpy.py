#log of modification cre

import json
import cv2
import math
import numpy as np
import os
import random 
from zipfile import ZipFile
import scipy.signal as signal
import pandas as pd
from generator_contour import Save_Contour_pkl

file_dir   = "D:/PhD/trying/tradition_method/stastics_backup/10.json"

with open(file_dir) as f:
    data = json.load(f)

# Output: {'name': 'Bob', 'languages': ['English', 'Fench']}
print(data)
shape  = data["shapes"]
print(shape)
num_line  = len(shape)
coordinate  = shape[0]["points"]
print(coordinate)










