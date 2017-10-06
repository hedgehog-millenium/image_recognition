import numpy as np
import cv2
import matplotlib.pyplot as plt
from imageHelper import image_helper as ih

YELLOW = 30
BLUE = 210
GREEN = 145
RED = 320


def find_black_dots(img_gray):
    lower = (0, 0, 0)
    upper = (0, 0, 0)
    img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    # black_mask = cv2.inRange(img,lower,upper)
    # circles = cv2.HoughCircles(black_mask,cv2.HOUGH_GRADIENT,1,20,param1=20,param2=8,
    #                             minRadius=0,maxRadius=60)

    imagem = cv2.bitwise_not(img)
    ih.show_imgs_cv2([img, imagem])


img_origin = cv2.imread('images/whater2.jpg', 1)
img_md = cv2.resize(img_origin, (0, 0), fx=0.25, fy=0.25)

img_gray = cv2.cvtColor(img_md, cv2.COLOR_BGR2GRAY)
img_hsv = cv2.cvtColor(img_md, cv2.COLOR_BGR2HSV)

# hsc hue sat value
# lower = np.array([0,0,0])
# upper = np.array([255,185,255])

# define range of blue color in HSV
lower_blue = np.array([0, 183, 91])
upper_blue = np.array([255, 255, 255])

lower_blue_nb = np.array([0, 84, 0])
upper_blue_nb = np.array([255, 255, 255])

_, threshold = cv2.threshold(img_md, 20, 255, cv2.THRESH_BINARY)
mask = cv2.inRange(img_hsv, lower_blue_nb, upper_blue_nb)

blur_gray_mask = cv2.medianBlur(mask, 19)  # smooth image by 21x21 pixels, may need to adjust a bit

# _, contours, _ = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
# cv2.drawContours(img_md, contours, -1, (0,0,255), 3)


# find_black_dots(blur_gray)

imgHelper = ih(img_md)
imgHelper.play_with_hsv_and_blur()
# ih.show_imgs_cv2([img_md, mask,blur_gray_mask ])
# ih.play_with_blur(mask)

# imgHelper.show_imgs_plt(img_gray)





#
# print(img_hsv.shape )
# h, s, v = cv2.split(img_hsv)
# print(np.max(h))
# print(np.max(s))
# print(np.max(v))

# fig = plt.figure()
# fig.add_subplot(2,2,1).imshow(img,cmap='gray')
# fig.add_subplot(2,2,2).imshow(img_bottle, cmap='gray')
# fig.add_subplot(2,2,3).imshow(img_bottle, cmap='gray')
# fig.add_subplot(2,2,4).imshow(img_bottle, cmap='gray')
# plt.show()
