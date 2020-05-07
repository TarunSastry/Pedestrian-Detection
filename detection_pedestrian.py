import cv2
import numpy as np

video_src = 'sample.mp4'

capture = cv2.VideoCapture(video_src)

people_cascade = cv2.CascadeClassifier('pedestrian.xml')

while True:
    rectangle, image = capture.read()
	
    
    if (type(image) == type(None)):
        break
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    pedastrian = people_cascade.detectMultiScale(gray,1.3,2)

    for(a,b,c,d) in pedastrian:
        cv2.rectangle(image,(a,b),(a+c,b+d),(0,255,210),4)
    
    cv2.imshow('video', image)
    
    if cv2.waitKey(33) == 27:
        break

cv2.destroyAllWindows()
