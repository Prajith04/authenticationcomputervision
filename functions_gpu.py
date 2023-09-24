from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.models import Model, Sequential
from keras.layers import  Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dropout, Activation
from keras.layers import Flatten, Dense
import tensorflow as tf
from keras.applications.mobilenet_v2 import preprocess_input
import cv2
import numpy as np
import mediapipe as mp
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()
def extract_face_features(image):
    # Convert the image to RGB color space    image=cv2.imread(image)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #image = (image * 255).astype(np.uint8)

    # Detect the face mesh
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if results.multi_face_landmarks:
        # Extract the landmark coordinates of the first detected face
        landmarks = results.multi_face_landmarks[0].landmark

        # Convert the landmark coordinates to a NumPy array
        features = np.array([[landmark.x, landmark.y, landmark.z] for landmark in landmarks])

        # Flatten the array and return the features
        return features.flatten()
model_mobilenet = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False)
# Initialize MediaPipe Hands model
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8, min_tracking_confidence=0.6)
def image_preprocess(image_path):
    img =cv2.imread(image_path)
    resize_img =cv2.resize(img,(224, 224))
    array_img = image.img_to_array(resize_img)
    expand_img = np.expand_dims(array_img, axis=0)
    preprocess_img = preprocess_input(expand_img)
    return preprocess_img
def extract_features(img):
    # Load and preprocess image
    img = img.img_to_array(img)
    img = preprocess_input(img)
    img = tf.expand_dims(img, axis=0)

    # Extract features using MobileNetV2
    features = model_mobilenet.predict(img)
    features = tf.reduce_mean(features, axis=(1, 2))

    features.numpy()
    features.Flatten()

def findCosineSimilarity(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation),(test_representation))
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))
def findEuclideanDistance(source_representation, test_representation):
    euclidean_distance = np.sqrt(np.sum(np.square(source_representation - test_representation)))
    return euclidean_distance
def facemodel():
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Convolution2D(4096, (7, 7), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(4096, (1, 1), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(2622, (1, 1)))
    model.add(Flatten())
    model.add(Activation('softmax'))
    model.load_weights("C:\\Users\\praji\\vgg_face_des.h5")      
    vgg_face_des = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)
    return vgg_face_des
def extract_hand_features(image):
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0].landmark
            features = np.array([[landmark.x, landmark.y, landmark.z] for landmark in landmarks])
            return features.flatten()


