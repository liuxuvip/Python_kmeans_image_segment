import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from tqdm import tqdm
block_size = 5
img = cv2.imread('./im01.png')#image read be 'gray'
fig=plt.figure(1)
plt.subplot(121),plt.imshow(img),plt.title('original')
plt.xticks([]),plt.yticks([])

#change img(2D) to 1D

# img = img.reshape((img.shape[0]*img.shape[1],3))
# img = np.float32(img)

simg_zeroize=np.zeros([img.shape[0]+block_size,img.shape[1]+block_size, img.shape[2]])
simg_zeroize[block_size/2:block_size/2+img.shape[0],block_size/2:block_size/2+img.shape[1],:] = img
immm = []
for j in tqdm(range(img.shape[0])):
    for k in range(img.shape[1]):
        dd1 = simg_zeroize[j:j+block_size,k:k+block_size,:]
        mg1 = dd1.reshape((block_size*block_size*img.shape[2]))
        immm.append(mg1)
immm = np.float32(immm)


#define criteria = (type,max_iter,epsilon)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,10,1.0)

#set flags: hou to choose the initial center
#---cv2.KMEANS_PP_CENTERS ; cv2.KMEANS_RANDOM_CENTERS
flags = cv2.KMEANS_RANDOM_CENTERS
# apply kmenas
compactness,labels,centers = cv2.kmeans(immm,20,None,criteria,10,flags)

img2 = labels.reshape((img.shape[0],img.shape[1]))

# io.imshow(img2)
# plt.show()
plt.subplot(122),plt.imshow(img2,'gray'),plt.title('kmeans')
plt.xticks([]),plt.yticks([])
plt.show()
io.imsave('./km.tif', img2.astype('uint8'))
print 'ok!'
