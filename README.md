# Embedded-night-vision-system-for-pedestrian-detection

## Introduction
Drivers can access a multitude of information from modern sensors, such as upcoming traffic lights, detours, traffic conditions, and much more. These sensors might not be able to accurately detect pedestrians or other objects at night, in low light conditions, or when the cameras are of poor quality. To solve this issue, we can make use of this app, which can help drivers spot pedestrians in the nighttime shadows and prevent accidents.

## HAAR HOG descriptors
### HAAR cascade
Objects can be located using the Haar cascade method, regardless of where they are in the image or how big they are. This algorithm is not unduly sophisticated and can run in real-time. A haar-cascade detector can be taught to recognise many different objects, such as cars, bikes, buildings, fruits, etc.
### HOG descriptor
The histogram of directed gradients is a feature descriptor for object detection in computer vision and image processing (HOG). The technique counts the instances of a gradient orientation occurring in particular regions of an image. This method is similar to edge orientation histograms, scale-invariant feature transform descriptors, and shape contexts, but it differs in that it is computed on a dense grid of uniformly spaced cells and uses overlapping local contrast normalisation for more accuracy.

## Model performance
In this application, I will use 6 night vision photographs, and YOLOV2 and ADABOOST will be able to identify pedestrians from 4 of the images and all 6 of the images, respectively. However, ADABOOST will have a detection accuracy of 80% while YOLOV2 will have a detection accuracy of 4/6 or 0.66.<br/>
```ADABOOST detection rate = 6/6 * 100 = 100% - 20 (for false detection rate)  = 80%```<br/>
```YOLOV2 = 4/6 * 100 = 66%```
