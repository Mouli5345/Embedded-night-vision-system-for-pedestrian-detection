from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import numpy as np
from tkinter.filedialog import askopenfilename
import numpy as np 
import os
import cv2
import imutils
from yoloDetection import detectObject, displayImage

main = tkinter.Tk()
main.title("Embedded Night-Vision System for Pedestrian Detection")
main.geometry("1300x1200")

global filename
global model
global width
global height
all_weights = []

class_labels = open('yolov2model/yolov2-labels').read().strip().split('\n') #reading labels from yolov2 model
cnn_model = cv2.dnn.readNetFromDarknet('yolov2model/yolov2.cfg', 'yolov2model/yolov2.weights') #reading model
cnn_layer_names = cnn_model.getLayerNames() #getting layers from cnn model
cnn_layer_names = [cnn_layer_names[i[0] - 1] for i in cnn_model.getUnconnectedOutLayers()] #assigning all layers


def detectFromImage(imagename): #function to detect object from images
    #random colors to assign unique color to each label
    label_colors = np.random.randint(0,255,size=(len(class_labels),3),dtype='uint8')
    try:
        image = cv2.imread(imagename) #image reading
        image_height, image_width = image.shape[:2] #converting image to two dimensional array
    except:
        raise 'Invalid image path'
    finally:
        image, _, _, _, _ = detectObject(cnn_model, cnn_layer_names, image_height, image_width, image, label_colors, class_labels)#calling detection function
    return image        



def uploadImage():
    global filename
    global width
    global height
    filename = filedialog.askopenfilename(initialdir="testImages")
    pathlabel.config(text=filename+" image loaded")
    img = cv2.imread(filename)
    cv2.imshow('Original Uploaded Image', img)
    cv2.waitKey();    
    


def yoloDetect():
    global filename
    image = cv2.imread(filename)
    detect = detectFromImage(filename)
    cv2.imshow("Original Image",image)
    cv2.imshow("Pedestrian Detected Image with YOLOV2",detect)    
    cv2.waitKey();

    
def adBoost(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def haarDetect(image,img1):
    hog = cv2.HOGDescriptor() 
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector()) 
    image = imutils.resize(image, width=min(800, 800))
    img1 = imutils.resize(img1, width=min(800, 800))
    (regions, _) = hog.detectMultiScale(image, winStride=(4, 4), padding=(8, 8), scale=1.05)    
    for (x, y, w, h) in regions:
        if (y+h) > 350:
            cv2.rectangle(img1, (x, y),  (x + w, y + h),  (0, 0, 255), 2)
            print(str(y)+" "+str(h)+" "+str((y+h)))
    return img1  

def adaboostDetect():
    global filename
    image = cv2.imread(filename)
    adjusted = adBoost(image, gamma=3.5)
    cv2.imwrite("temp.png",adjusted)
    img= cv2.imread(filename)
    img1 = cv2.imread('temp.png')
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray_img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    gray_img_eqhist = cv2.equalizeHist(gray_img)
    gray_img1_eqhist = cv2.equalizeHist(gray_img1)
    clahe = cv2.createCLAHE(clipLimit=20)
    gray_img_clahe = clahe.apply(gray_img_eqhist)
    th=80
    max_val=255
    ret, o3 = cv2.threshold(gray_img_clahe, th, max_val, cv2.THRESH_TOZERO)
    images = haarDetect(o3,img1)
    cv2.imshow("Original Image",image)
    cv2.imshow("Pedestrian Detected Image with HAAR + AdaBoost",images)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
        
    
        

def exit():
    main.destroy()
    

    
font = ('times', 16, 'bold')
title = Label(main, text='Embedded Night-Vision System for Pedestrian Detection')
title.config(bg='yellow4', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)


font1 = ('times', 14, 'bold')
upload = Button(main, text="Upload Night Vision Image", command=uploadImage)
upload.place(x=50,y=100)
upload.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='yellow4', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=50,y=150)

cnnButton = Button(main, text="Night Vision Pedestrain Detection using YOLOV2", command=yoloDetect)
cnnButton.place(x=50,y=200)
cnnButton.config(font=font1)

fusionButton = Button(main, text="Night Vision Pedestrain Detection using HAAR + AdaBoost", command=adaboostDetect)
fusionButton.place(x=50,y=250)
fusionButton.config(font=font1)

exitButton = Button(main, text="Exit", command=exit)
exitButton.place(x=50,y=300)
exitButton.config(font=font1)


main.config(bg='magenta3')
main.mainloop()
