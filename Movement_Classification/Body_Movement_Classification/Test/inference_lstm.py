import cv2
import mediapipe as mp
import numpy as np
import threading
import tensorflow as tf
import os
import csv
import time

label = ""  # Initialize label as an empty string
n_time_steps = 2
lm_list = []

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

model = tf.keras.models.load_model("/home/prolab/Schreibtisch/Neue_Masterarbeit/Movement_Classification/Body_Movement_Classification/Train/model.h5")

cap = cv2.VideoCapture('/home/prolab/Schreibtisch/Neue_Masterarbeit/Dataset/Videos_own_Dataset/C0028.MP4')

fps = cap.get(cv2.CAP_PROP_FPS)
print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
print(f"Video FPS: {fps}")
print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

def make_landmark_timestep(results):
    c_lm = []
    for id, lm in enumerate(results.pose_landmarks.landmark):
        c_lm.append(lm.x)
        c_lm.append(lm.y)
        c_lm.append(lm.z)
        c_lm.append(lm.visibility)
    return c_lm

def draw_landmark_on_image(mpDraw, results, img):
    mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
    for id, lm in enumerate(results.pose_landmarks.landmark):
        h, w, c = img.shape
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
    return img

def draw_class_on_image(label, img):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 30)
    fontScale = 1
    fontColor = (0, 255, 0)
    thickness = 2
    lineType = 2
    cv2.putText(img, label,
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                thickness,
                lineType)
    return img

def detect(model, lm_list, label_mapping):
    global label
    
    lm_list = np.array(lm_list)
    lm_list = np.expand_dims(lm_list, axis=0)
    print(lm_list.shape)
    
    results = model.predict(lm_list)
    print(results)
    
    # Um die Vorhersagewerte in Prozent zu erhalten
    predicted_probabilities = tf.nn.softmax(results[0])
    
    # Anzeigen der Vorhersagewerte in Prozent
    for i, prob in enumerate(predicted_probabilities):
        class_name = label_mapping[i]
        percentage = round(prob.numpy() * 100, 2)
        # print(f"{class_name}: {percentage}%")
    
    predicted_label_id = np.argmax(results, axis=1)[0]
    # print(predicted_label_id)
    predicted_label = label_mapping[predicted_label_id]
    
    # # # Überprüfe die Erkennungsgenauigkeit und setze das Label entsprechend
    # if predicted_probabilities[predicted_label_id] < 0.6:
    #     predicted_label = "negativ"
    
    print(predicted_label)
    
    # Aktualisiere das globale Label
    label = predicted_label
    
    return label

# Define label mapping  
label_mapping = {0: "Neutral", 1: "AP_RH_ST", 2: "AP_LH_ST", 3: "AP_BH_ST"}

# CSV-Datei initialisieren
with open('predictions.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Frame_Number', 'Predictions'])

i = 0

while True:
    start_time = time.time()

    success, img = cap.read()
    if not success:
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    i = i + 1

    # Remove the print statement
    # print("Start detect....")

    if results.pose_landmarks:
        c_lm = make_landmark_timestep(results)

        lm_list.append(c_lm)
        if len(lm_list) == n_time_steps:
            # predict
            t1 = threading.Thread(target=detect, args=(model, lm_list, label_mapping,))
            t1.start()
            lm_list = []

            # Save predictions to CSV
            predictions = label
            frame_number = i
            with open('predictions.csv', mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([frame_number, predictions])

        img = draw_landmark_on_image(mpDraw, results, img)

    img = draw_class_on_image(label, img)
    cv2.imshow("Image", img)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
