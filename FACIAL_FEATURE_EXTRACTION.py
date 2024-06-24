import cv2
import dlib
import imutils
import numpy as np
from collections import deque
from imutils import face_utils
from scipy.spatial import distance as dist
from sklearn.neighbors import KNeighborsClassifier
import tensorflow as tf
from tensorflow import keras 
import os
import json

# Constants 
prediction_window = deque(maxlen=10)
drowsy_threshold = 0.7
FLP = "shape_predictor_68_face_landmarks.dat"
faceFinder = dlib.get_frontal_face_detector()
model = keras.models.load_model('1000epochs.h5')
featureFinder = dlib.shape_predictor(FLP)
cameraFrame = cv2.VideoCapture(0)

# Facial features: 
(leftEyeStart, leftEyeEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rightEyeStart, rightEyeEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mouthStart, mouthEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

# 3D point abstraction: 
model_points = np.array([
    (0.0, 0.0, 0.0), # Tip of the nose            
    (0.0, -330.0, -65.0),  # Chin   
    (-225.0, 170.0, -135.0), # Left corner of the left eye   
    (225.0, 170.0, -135.0), # Right corner of the right eye     
    (-150.0, -150.0, -125.0), #Left corner of the mouth
    (150.0, -150.0, -125.0) # Right corner of the mouth      
], dtype=np.float64)

# Facial Analysis from feature extraction: 

def ear_calculation(eye):
    p1_to_p4 = dist.euclidean(eye[0], eye[3])
    p2_to_p6 = dist.euclidean(eye[1], eye[5])
    p3_to_p5 = dist.euclidean(eye[2], eye[4])
    ear = (p2_to_p6 + p3_to_p5) / (2.0 * p1_to_p4)
    return ear

def mar_calculation(mouth):
    p1_to_p4 = dist.euclidean(mouth[2], mouth[10])
    p2_to_p6= dist.euclidean(mouth[3], mouth[8])
    p3_to_p5 = dist.euclidean(mouth[0], mouth[6])
    mar = (p1_to_p4 + p2_to_p6) / (2 * p3_to_p5)
    return mar

def calculate_angles(image_points2d):
    size = image.shape
    focal_length = size[1]
    center = (size[1]/2, size[0]/2)

    camera_matrix = np.array(
                    [[focal_length, 0, center[0]],
                    [0, focal_length, center[1]],
                    [0, 0, 1]], dtype="double"
                    )
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points2d, camera_matrix, dist_coeffs)
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    theta_x = np.arctan2(rotation_matrix[2][1], rotation_matrix[2][2])
    theta_y = np.arctan2(-rotation_matrix[2][0], np.sqrt(rotation_matrix[2][1] ** 2 + rotation_matrix[2][2] ** 2))
    theta_x_deg = np.degrees(theta_x) * -1
    theta_y_deg = np.degrees(theta_y) * -1
    (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    return (theta_x_deg, theta_y_deg, nose_end_point2D)

# Function to determine if the majority of the last predictions are drowsy
def is_drowsy():
    if prediction_window.count(1) / len(prediction_window) >= drowsy_threshold:
        return True
    else:
        return False

# Update the prediction window with the latest prediction
def update_prediction_window(prediction):
    if prediction > drowsy_threshold:
        prediction_window.append(1)
    else:
        prediction_window.append(0)

def extract_features(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceFinder(gray_image, 0)
    if len(faces) == 0:
        # No faces detected, return None
        return None

    for face in faces:
        # Process each detected face
        face_landmarks = featureFinder(gray_image, face)
        face_landmarks = face_utils.shape_to_np(face_landmarks)
        left_eye = face_landmarks[leftEyeStart:leftEyeEnd]
        right_eye = face_landmarks[rightEyeStart:rightEyeEnd]
        mouth = face_landmarks[mouthStart:mouthEnd]
        EAR_left = ear_calculation(left_eye)
        EAR_right = ear_calculation(right_eye)
        MAR = mar_calculation(mouth)
        ear = (EAR_left + EAR_right) / 2.0
        image_points2d = np.array([
            (face_landmarks[30][0], face_landmarks[30][1]),  # Nose
            (face_landmarks[8][0], face_landmarks[8][1]),    # Chin
            (face_landmarks[45][0], face_landmarks[45][1]),  # Left eye left corner
            (face_landmarks[36][0], face_landmarks[36][1]),  # Right eye Right corner
            (face_landmarks[64][0], face_landmarks[64][1]),  # left mouth corner
            (face_landmarks[60][0], face_landmarks[60][1]),  # right mouth corner
        ], dtype=np.float64)
        (theta_x_deg, theta_y_deg, nose_end_point2D) = calculate_angles(image_points2d)
        for p in image_points2d:
            cv2.circle(image, (int(p[0]), int(p[1])), 3, (0,0,255), -1)
        p1 = ( int(image_points2d[0][0]), int(image_points2d[0][1]))
        p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
        cv2.line(image, p1, p2, (255,0,0), 2)
        input_features = np.array([[ear, MAR, theta_x_deg]])        
        prediction = model.predict(input_features)
        print("prediction: " + str(prediction))
        update_prediction_window(prediction)
        if is_drowsy():
            cv2.putText(image, "DROWSY!", (20, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 2)
 

        
        cv2.putText(image, "Eye aspect ratio: {}".format(round(ear, 3)), (20, 200), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    1, (255, 255, 255), 2)
        cv2.putText(image, "Mouth aspect ratio: {}".format(round(MAR, 3)), (20, 300), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    1, (255, 255, 255), 2)
        cv2.putText(image, "Verticle angle: {}".format(round(theta_x_deg, 3)), (20, 400), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    1, (255, 255, 255), 2)
        return [ear, MAR, theta_x_deg]
    return None

def train_data(folder_path):
    results = []
    i = 0
    for filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, filename)
        image = cv2.imread(image_path)
        image = imutils.resize(image, width=800)
        results.append(extract_features(image))
        print (i)
        i += 1
    output_data_path = "nondrowsy.json"
    with open(output_data_path, "w") as json_file:
        json.dump(results, json_file)

try:
    response = input("Waiting for input. For live Input press 1. For stored input press 2: ")
    while True:
        if response == "1":
            (status, image) = cameraFrame.read()
            image = imutils.resize(image, width=800)
            extract_features(image)
            cv2.imshow("Frame", image)
            if not status:
                print("Error: Unable to read frame from the camera.")
                break
        elif response == "2":
            train_data("C:/Users/benja/Desktop/Driver Drowsiness Dataset (DDD)/Non Drowsy")
        else:
            print("Error: Invalid input.")
            break
        key = cv2.waitKey(1)
        if key == ord("q"):
            break
finally:
    # Release the camera and close all OpenCV windows
    cameraFrame.release()
    cv2.destroyAllWindows()
