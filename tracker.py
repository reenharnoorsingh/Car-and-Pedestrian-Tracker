# Car and Pedestrian Tracker

import cv2  # pip install opencv-python

image_file = 'car.jpg'  # car image check repository for the image

classifier_file = 'car_detector.xml'  # pre-trained car detection algorithm

img = cv2.imread(image_file)  # create opencv image
grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   # makes the image black and white

cartracker = cv2.CascadeClassifier(classifier_file) #create car classifier

cars = cartracker.detectMultiScale(grayscale) #detect cars

for (x,y,w,h) in cars:
    cv2.rectangle(img, (x,y),(x+w , y+h),(0,0,225))

cv2.imshow("Car Detector", grayscale)  # display image with car spotted


cv2.waitKey()  # don't autoclose
