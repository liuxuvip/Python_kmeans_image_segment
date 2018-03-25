import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import io


img = cv2.imread('./im01.png')#image read be 'gray'
# plt.subplot(121),plt.imshow(img,'gray'),plt.title('original')
fig=plt.figure(1)
plt.subplot(121),plt.imshow(img),plt.title('original')
plt.xticks([]),plt.yticks([])

#change img(2D) to 1D

img1 = img.reshape((img.shape[0]*img.shape[1],3))
img1 = np.float32(img1)

#define criteria = (type,max_iter,epsilon)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,10,1.0)

#set flags: hou to choose the initial center
#---cv2.KMEANS_PP_CENTERS ; cv2.KMEANS_RANDOM_CENTERS
flags = cv2.KMEANS_RANDOM_CENTERS
# apply kmenas
compactness,labels,centers = cv2.kmeans(img1,20,None,criteria,10,flags)

img2 = labels.reshape((img.shape[0],img.shape[1]))

plt.subplot(122),plt.imshow(img2,'gray'),plt.title('kmeans')
plt.xticks([]),plt.yticks([])
plt.show()
io.imsave('./km.tif', img2.astype('uint8'))
