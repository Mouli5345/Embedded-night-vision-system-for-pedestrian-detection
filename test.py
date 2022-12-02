import cv2
import numpy as np
import imutils
from PIL import Image

def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)
'''
def getImage(image):
    hog = cv2.HOGDescriptor() 
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector()) 
    image = imutils.resize(image, width=min(400, image.shape[1]))    
    (regions, _) = hog.detectMultiScale(image, winStride=(4, 4), padding=(8, 8), scale=1.05)    
    for (x, y, w, h) in regions:
        cv2.rectangle(image, (x, y),  (x + w, y + h),  (0, 0, 255), 2)
    return image    

image = cv2.imread('test.png')
adjusted = adjust_gamma(image, gamma=3.5)


image = getImage(image)
adjusted = getImage(adjusted)
cv2.imwrite("temp.png",adjusted)

#cv2.imshow("Image", image)
cv2.imshow("adjust", adjusted) 
cv2.waitKey(0) 
   
cv2.destroyAllWindows()
'''
def getImage(image,img1):
    hog = cv2.HOGDescriptor() 
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector()) 
    image = imutils.resize(image, width=min(800, 800))
    img1 = imutils.resize(img1, width=min(800, 800))
    (regions, _) = hog.detectMultiScale(image, winStride=(4, 4), padding=(8, 8), scale=1.05)    
    for (x, y, w, h) in regions:
        cv2.rectangle(img1, (x, y),  (x + w, y + h),  (0, 0, 255), 2)
        print(str(x)+" "+str(y))
    return img1
image = cv2.imread('5.png')
adjusted = adjust_gamma(image, gamma=3.5)
cv2.imwrite("temp.png",adjusted)
img= cv2.imread('5.png')
img1=cv2.imread('temp.png')
gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray_img1=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)

gray_img_eqhist=cv2.equalizeHist(gray_img)
gray_img1_eqhist=cv2.equalizeHist(gray_img1)
clahe=cv2.createCLAHE(clipLimit=20)
gray_img_clahe=clahe.apply(gray_img_eqhist)
'''
gray_img1_clahe=clahe.apply(gray_img1_eqhist)
images=np.concatenate((gray_img_clahe,gray_img1_clahe),axis=1)
images = getImage(images)
cv2.imshow("Images",images)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

th=80
max_val=255
ret, o3 = cv2.threshold(gray_img_clahe, th, max_val, cv2.THRESH_TOZERO)
images = getImage(o3,img1)
cv2.imshow("Images",images)
cv2.waitKey(0)
cv2.destroyAllWindows()
