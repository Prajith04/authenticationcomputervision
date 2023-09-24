import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing import image
from keras.utils import img_to_array
from keras.applications.imagenet_utils import preprocess_input
from functions_gpu import *
import pickle
import mediapipe as mp
from keras.models import load_model
import time
pTime = 0
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
# Initialize the HandTrack model_face_detect
hands = mp_hands.Hands(static_image_mode=False,max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
gesture_names = ['0','1','2','3','4','5','6','7','8','9']

# Load face recognition model_face_detect and faces database
model_face_recognize=facemodel()
filename = 'faces.pkl'
with open(filename, 'rb') as f:
    faces = pickle.load(f)

# Load saved classifier from file
with open('hand_gesture_classifier.pkl', 'rb') as f:
    clf = pickle.load(f)

# Load face detection model_face_detect
face_detect = cv2.CascadeClassifier("C:\\Users\\praji\\anaconda3\\pkgs\\opencv-4.6.0-py310ha7641e4_2\\Library\\etc\\haarcascades\\haarcascade_frontalface_default.xml")
model_face_detect = cv2.dnn.readNetFromCaffe('C:\\Users\\praji\\deploy.prototxt', 'C:\\Users\\praji\\res10_300x300_ssd_iter_140000.caffemodel')

# Initialize video capture
cam_vid = cv2.VideoCapture(0)

entered_pin=[]
temp_pin=[]
previous_gesture = None
while(True):
    # Read frame from video capture
    ret,frame = cam_vid.read()
    if ret==False:
        break
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(frame, f'FPS:{int(fps)}', (500, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    height,width,channels=frame.shape
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300,300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    model_face_detect.setInput(blob)
    detections = model_face_detect.forward()
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([width, height,width,height])
            (x, y, w, h) = box.astype("int")
            cv2.rectangle(frame, (x, y), (w, h), (0, 0, 255), 2)
            crop_face = frame[y:h, x:w]
            if crop_face is not None and crop_face.shape[0] > 0 and crop_face.shape[1] > 0:
                    resize_img =cv2.resize(crop_face,(224, 224))
                    array_img = img_to_array(resize_img)
                    expand_img = np.expand_dims(array_img, axis = 0)
                    expand_img/=127
                    expand_img-=1
                    hand_features = extract_hand_features(frame)
                    original_pin=['1','2','7','8']
                    
                # Use the normalized landmarks as input to the pre-built gesture recognition model_face_detect
        # Check if the face is large enough for recognition
        
                    
                    
                
                    
                    captured_representation =model_face_recognize.predict(expand_img)[0,:]
                    found = 0
                    line_x1 = x
                    line_y1 = int((y + h) / 2)
                    line_x2 = int(x + (w - x) / 2)
                    line_y2 = line_y1
                    for i in faces:
                        name = i
                        representation =faces[i]
                        similarity = findCosineSimilarity(representation,captured_representation)
                        if(similarity<0.17):
                            cv2.putText(frame,'PRAJITH', (int(w+10), int(y-12)), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 255, 0), 2)
                            if hand_features is not None:
                                hand_features=hand_features.reshape(1,-1)
                                predicted_class = clf.predict(hand_features)[0]
                                gesture_name = gesture_names[predicted_class]
                                cv2.putText(frame,gesture_name , (50, 75), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0, 0), 2)
                                if previous_gesture != gesture_name:
                                    previous_gesture = gesture_name
                                    temp_pin += str(predicted_class)
                                    if len(entered_pin) < 4 :
                                        entered_pin += str(predicted_class)
                                        cv2.putText(frame,f'Digit Entered:{gesture_name}' , (50,150), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0, 0), 2)
                            if entered_pin==original_pin:
                                cv2.putText(frame,'Welcome back!', (int(w+10), int(y-50)), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0, 255, 0), 2)
                            elif entered_pin!=original_pin and len(entered_pin)==len(original_pin):
                                cv2.putText(frame,'Try agian', (int(w+10), int(y-50)), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,0,255), 2)
                            else:
                                cv2.putText(frame,'waiting for verification', (int(w+10), int(y-50)), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,0, 0), 2)

                            cv2.line(frame,(int((x+w)/2),y+15),(w-5,y-20),(0, 255, 0),1)
                            cv2.line(frame,(x+w,y+10),(x+w+10,y+10),(0, 255, 0),1)
                            found = 1
                            break
# Draw line from rectangle to name
                    
                    if(found == 0): 
                        cv2.putText(frame, 'STRANGER', (int(w+10), int(y-12)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0, 255), 2)
                        cv2.putText(frame, 'Access denied!', (int(w+10), int(y-50)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cv2.line(frame,(int((x+w)/2),y+15),(w-5,y-20),(0, 255, 0),1)
                        cv2.line(frame,(x+w,y+10),(x+w+10,y+10),(0, 255, 0),1)
    cv2.imshow('img',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break  
print(entered_pin)
cam_vid.release()
cv2.destroyAllWindows()
