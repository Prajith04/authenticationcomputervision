import pickle
import os
from functions_gpu import *
import numpy as np
import cv2
face_detect = cv2.CascadeClassifier("C:\\Users\\praji\\anaconda3\\pkgs\\opencv-4.6.0-py310ha7641e4_2\\Library\\etc\\haarcascades\\haarcascade_frontalface_default.xml")
model_face_detect = cv2.dnn.readNetFromCaffe('C:\\Users\\praji\\deploy.prototxt', 'C:\\Users\\praji\\res10_300x300_ssd_iter_140000.caffemodel')
model=facemodel()
filename = 'faces.pkl'
faces_directory="C:\\Users\\praji\\OneDrive\\Pictures\\faces"
faces=dict()
face_cam=cv2.VideoCapture(0)
for count in range(100): 
    ret,frame=face_cam.read()
    if ret==False:
        break
    height,width,channels=frame.shape
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    model_face_detect.setInput(blob)
    detections = model_face_detect.forward()
        
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([width, height,width,height])
            (x, y, w, h) = box.astype("int")
            cv2.rectangle(frame, (x, y), (w, h), (0, 0, 255), 2)
            crop_face = frame[y:h, x:w]
            cv2.imwrite(os.path.join(faces_directory, f'face_{len(os.listdir(os.path.join(faces_directory)))}.jpg'), crop_face)
    #verifying detection
    cv2.imshow('video',frame)
    cv2.waitKey(1)    
#closing window and flushing used video 
face_cam.release()
cv2.destroyAllWindows()    
for image in os.listdir(faces_directory):
    face, extension = image.split(".")
    faces[face] = model.predict(image_preprocess(os.path.join(faces_directory,image)))[0,:]
with open(filename, 'wb') as f:
    pickle.dump(faces, f)