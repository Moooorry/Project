from cv2 import cv2 
import numpy as np
import matplotlib.pyplot as plt

def edge_detection():
    img = cv2.imread('2.jpg',cv2.IMREAD_GRAYSCALE)
    lap = cv2.Laplacian(img,cv2.CV_64F,ksize=3)
    lap = np.uint8(np.absolute(lap))
    sobelX = cv2.Sobel(img,cv2.CV_64F,1,0)
    sobelY = cv2.Sobel(img,cv2.CV_64F,0,1)

    sobelX = np.uint8(np.absolute(sobelX))
    sobelY = np.uint8(np.absolute(sobelY))

    sobelCombined = cv2.bitwise_or(sobelX,sobelY)

    titles = ['image','Laplacian','sobelX','sobelY','sobelCombined']
    images = [img,lap,sobelX,sobelY,sobelCombined]
    for i in range(5):
        plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])

    plt.show()

def candy_edge_detection():
    img = cv2.imread('2.jpg',0)
    canny = cv2.Canny(img,50,100)

    titles = ['image','canny']
    images = [img,canny]
    for i in range(2):
        plt.subplot(1,2,i+1),plt.imshow(images[i],'gray')
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])

    plt.show()

def grab_cut():
    img = cv2.imread('3.jpg')
    mask = np.zeros(img.shape[:2],np.uint8)

    bgModel = np.zeros((1,65),np.float64)
    fgModel = np.zeros((1,65),np.float64)

    rect = (420,1,676,881)

    cv2.grabCut(img,mask,rect,bgModel,fgModel,5,cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img = img*mask2[:,:,np.newaxis]

    plt.subplot(121)
    plt.title('Grabcut')
    plt.xticks([]),plt.yticks([])
    plt.imshow(img)
    plt.subplot(122)
    plt.imshow(cv2.cvtColor(cv2.imread('3.jpg'),cv2.COLOR_BGR2RGB))
    plt.title('Original')
    plt.xticks([]),plt.yticks([])
    plt.show()


edge_detection()
candy_edge_detection()
grab_cut()


