import os
import cv2
import numpy as np
import mediapipe as mp
from sklearn.svm import SVC
import pickle

# Initialize MediaPipe Hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.6)

# Define function to extract hand features from an image using MediaPipe Hands
def extract_hand_features(image):
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0].landmark
            features = np.array([[landmark.x, landmark.y, landmark.z] for landmark in landmarks])
            return features.flatten()
        else:
            return None

# Load dataset
data_dir = "C:\\Users\\praji\\Downloads\\Sign Language for Numbers"
classes = os.listdir(data_dir)
# Extract hand features from each image in dataset
X = []
y = []
for class_idx, class_name in enumerate(classes):
    class_dir = os.path.join(data_dir, class_name)
    for img_name in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_name)
        img = cv2.imread(img_path)
        img=cv2.resize(img,(100,100))
        hand_features = extract_hand_features(img)
        if hand_features is not None:
            X.append(hand_features)
            y.append(class_idx)
        cv2.imshow('video',img)
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break 
clf = SVC(kernel='linear', C=1, decision_function_shape='ovr')
clf.fit(X, y)
with open('hand_gesture_classifier1.pkl', 'wb') as f:
    pickle.dump(clf, f)
