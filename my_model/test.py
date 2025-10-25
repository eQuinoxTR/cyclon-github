import numpy as np
import cv2

cap = cv2.VideoCapture("https://192.168.0.195:8080/video")

while True:
    ret, frame = cap.read()

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()