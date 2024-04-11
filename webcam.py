import numpy as np
import cv2 as cv
import tensorflow as tf
import utils
from keras import models

cap = cv.VideoCapture(0)
model = models.load_model("checkpoints/model-binarize.bkp.keras")
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
upleft = int(width / 2 - height / 2)
upright = int(width / 2 + height / 2)
kernel = np.ones((3, 3), np.uint8)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv.flip(frame, 0)
    frame_resized = frame[0:height, upleft:upright]
    frame_resized = cv.resize(frame_resized, (128, 128))
    frame_resized = cv.cvtColor(frame_resized, cv.COLOR_BGR2GRAY)
    _, frame_resized = cv.threshold(frame_resized, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    # frame_resized = cv.dilate(frame_resized, kernel, iterations=1)
    frame_resized = cv.cvtColor(frame_resized, cv.COLOR_GRAY2BGR)
    # frame_resized = utils.fourier(frame_resized)
    
    cv.imshow('Processed', frame_resized)
    
    frame_resized = np.expand_dims(frame_resized, axis=0)  # Add batch dimension
    res = np.argmax(model.predict(frame_resized))
    
    cv.putText(frame, str(res), (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)
    cv.imshow('Webcam', frame)
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
