import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2
import uuid   # Unique identifier
import os
import time

model = torch.hub.load('ultralytics/yolov5', 'custom', path='/Users/shashanksaraswat/yolov5/best.pt', force_reload=True)


cap = cv2.VideoCapture('/Users/shashanksaraswat/Desktop/birdsandvideo.mpg')

while cap.isOpened():
    ret, frame = cap.read()

    # Make detections
    results = model(frame)

    cv2.imshow('Drone Detection Window', np.squeeze(results.render()))
    # plt.show('Drone Detection Window', np.squeeze(results.render()))
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()