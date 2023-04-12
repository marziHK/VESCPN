
import os
import cv2 as cv
from os.path import isfile, join
import matplotlib.pyplot as plt

PathImg = './results/cal/'
# PathImg = './videos/gt_wak_x3/'
savedir = './videos/'
# ####################################################################
# ######################  Make predicted video  ######################
# ####################################################################

# frame per second
fps = 8
frames = []
files = [f for f in os.listdir(PathImg) if isfile(join(PathImg, f))]
files.sort()

#for sorting the file names properly
# files.sort(key = lambda x: x[5:-4])

for i in range(len(files)):
    filename = PathImg+'' + files[i]
    # print(filename)
    img = cv.imread(filename)
    height, width ,ch= img.shape
    frmSize = (width,height)
    
    #inserting the frames into an image array
    frames.append(img)


out = cv.VideoWriter(savedir+'cal_model_37ep.avi',cv.VideoWriter_fourcc(*'DIVX'), fps, frmSize)
for i in range(len(frames)):
    im = frames[i]
    out.write(im)
out.release()