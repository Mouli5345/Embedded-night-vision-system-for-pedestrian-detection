# KnightVision

## Introduction
Modern sensors can provide drivers with a wealth of information, including impending traffic signals, detours, traffic conditions, and much more. However, at night, when it is dark or the cameras are of poor quality, these sensors may not be able to provide accurate information about pedestrians or other objects. To overcome this problem, we can use Knight Vision, an application which can be used to assist drivers in mkaing out pedestrians in the dark strets in the night and avoid accidents.

## HAAR HOG descriptors
### HAAR cascade
No matter where they are in the image or how big they are, objects can be found using the process known as the Haar cascade. This algorithm can operate in real-time and is not overly complex. A haar-cascade detector can be trained to recognise a variety of items, including automobiles, bikes, structures, fruits, etc.
### HOG descriptor
A feature descriptor for object detection in computer vision and image processing is the histogram of oriented gradients (HOG). The method records the number of times a gradient orientation appears in specific areas of an image. This approach is comparable to edge orientation histograms, scale-invariant feature transform descriptors, and shape contexts; however, it differs in that it is computed on a dense grid of uniformly spaced cells and makes use of overlapping local contrast normalisation for increased accuracy.

## Model performance
In this application, I will use 6 night vision photographs, and YOLOV2 and ADABOOST will be able to identify pedestrians from 4 of the images and all 6 of the images, respectively. However, ADABOOST will have a detection accuracy of 80% while YOLOV2 will have a detection accuracy of 4/6 or 0.66.<br/>
```ADABOOST detection rate = 6/6 * 100 = 100% - 20 (for false detection rate)  = 80%```<br/>
```YOLOV2 = 4/6 * 100 = 66%```
