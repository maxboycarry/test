import cv2
import numpy as np




template = cv2.imread('case.png',0)
w, h = template.shape[::-1]
video_capture = cv2.VideoCapture(0)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 200)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 200) #해상도

while True:
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if gray.shape[0]>template.shape[0] and gray.shape[1]>template.shape[1]:
        res = cv2.matchTemplate(gray,template,cv2.TM_CCOEFF_NORMED)
        threshold = 0.8
        loc = np.where( res >= threshold)
        for pt in zip(*loc[::-1]):
            cv2.rectangle(frame, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
            print(pt,pt[0],pt[1])
        cv2.imshow('orginal', frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

       
