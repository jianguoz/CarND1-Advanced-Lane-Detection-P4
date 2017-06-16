'''Writen by Jianguo Zhang, June 14, 2017.
   jzhan51@uic.edu
'''

import numpy as np
import cv2
import pickle
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
#% matplotlib inline

# Read in and make a list of all calibration images
images = glob.glob('./camera_cal/calibration*.jpg')

objpoints = [] # 3D points in real world space
imgpoints = [] # 2D points or corners in image plane

height, width = 6, 9

# Prepare object points, like (0,0,0), (1,0,0),......(8,5,0)
objp = np.zeros((height*width,3), np.float32)
objp[:,:2] = np.mgrid[0:width, 0:height].T.reshape(-1,2) # x, y cordinates

for idx, image in enumerate(images):
    img = cv2.imread(image)
    #plt.imshow(img)
    # Convert to grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (width,height), None)

    if ret==True:
        # Draw detected corners on an image
        cv2.drawChessboardCorners(img, (width,height), corners, ret)
        
        # Update points
        objpoints.append(objp)
        imgpoints.append(corners)
        
        # Save image  
        cv2.imwrite('./Corners_found/corners_found'+str(idx)+'.jpg', img)
        
        # Also provides some interval before reading next frame so that we can adjust 
        # our chess board in different direction
#         cv2.imshow('img', img)
#         cv2.waitKey(500)
        
# cv2.destroyAllWindows()


# Take calibration5 for example
img = cv2.imread('./camera_cal/calibration5.jpg')

# Image size (width, height)
img_size = (img.shape[1], img.shape[0])
        
# Calibrate cameras
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

# Undistorting image
dst = cv2.undistort(img, mtx, dist, None, mtx)

# Save undistorted images
cv2.imwrite('./output_img/camera_calibration_example.jpg',dst)

# Save the camera calibration result for later use
dist_pickle = {}
dist_pickle["mtx"] = mtx
dist_pickle["dist"] = dist
pickle.dump(dist_pickle, open("./camera_cal/camera_dist_pickle.p", "wb"))

# Visualize undistortion
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=30)
ax2.imshow(dst)
ax2.set_title('Undistorted Image', fontsize=30)