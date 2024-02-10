import cv2
import mediapipe as mp
import pandas as pd
import os
from tqdm import tqdm

def process_videos_in_folder(folder_path, label):
    mpPose = mp.solutions.pose
    mpHands = mp.solutions.hands
    pose = mpPose.Pose()
    hands = mpHands.Hands()
    mpDraw = mp.solutions.drawing_utils

    lm_list_pose = []
    lm_list_left_hand = []
    lm_list_right_hand = []

    def make_landmark_timestep(results):
        c_lm = []
        for id, lm in enumerate(results.pose_landmarks.landmark):
            c_lm.append(lm.x)
            c_lm.append(lm.y)
            c_lm.append(lm.z)
            c_lm.append(lm.visibility)
        return c_lm

    def make_hand_landmark_timestep(hand_landmarks):
        c_lm = []
        for lm in hand_landmarks.landmark:
            c_lm.append(lm.x)
            c_lm.append(lm.y)
            c_lm.append(lm.z)
        return c_lm

    def draw_landmark_on_image(mpDraw, results, img):
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        return img

    def draw_hand_landmarks_on_image(mpDraw, hand_landmarks, img):
        mpDraw.draw_landmarks(img, hand_landmarks, mpHands.HAND_CONNECTIONS)
        return img

    files = os.listdir(folder_path)
    total_videos = len(files)
    processed_videos = 0

    for video_file in tqdm(files, desc="Processing videos", total=total_videos):
        video_path = os.path.join(folder_path, video_file)
        cap = cv2.VideoCapture(video_path)
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame_count += 1

                frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results_pose = pose.process(frameRGB)
                results_hands = hands.process(frameRGB)

                if results_pose.pose_landmarks:
                    lm_pose = make_landmark_timestep(results_pose)
                    lm_list_pose.append(lm_pose)
                    frame = draw_landmark_on_image(mpDraw, results_pose, frame)

                if results_hands.multi_hand_landmarks:
                    for i, hand_landmarks in enumerate(results_hands.multi_hand_landmarks):
                        lm_hand = make_hand_landmark_timestep(hand_landmarks)
                        handedness = results_hands.multi_handedness[i].classification[0].label

                        if handedness == "Left":
                            lm_list_left_hand.append(lm_hand)
                        elif handedness == "Right":
                            lm_list_right_hand.append(lm_hand)
                        frame = draw_hand_landmarks_on_image(mpDraw, hand_landmarks, frame)

                    # Check if lm_list_left_hand is not empty before appending a list of zeros
                    if not lm_list_left_hand:
                        lm_list_left_hand.append([0.0] * (len(lm_list_right_hand[0]) if lm_list_right_hand else 21 * 3))

                cv2.imshow("image", frame)
                if cv2.waitKey(1) == ord('q'):  # Hier können Sie eine maximale Anzahl von Frames festlegen
                    break
            else:
                break

        df_pose = pd.DataFrame(lm_list_pose, columns=[f'pose_{i}' for i in range(len(lm_list_pose[0]))])
        
        # Before creating the DataFrame for the left hand
        if lm_list_left_hand:
            df_left_hand = pd.DataFrame(lm_list_left_hand, columns=[f'left_hand_{i}' for i in range(len(lm_list_left_hand[0]))])
        else:
            # If lm_list_left_hand is empty, create a DataFrame with zeros
            df_left_hand = pd.DataFrame(columns=[f'left_hand_{i}' for i in range(21 * 3)])

        # Before creating the DataFrame for the right hand
        if lm_list_right_hand:
            df_right_hand = pd.DataFrame(lm_list_right_hand, columns=[f'right_hand_{i}' for i in range(len(lm_list_right_hand[0]))])
        else:
            # If lm_list_right_hand is empty, create a DataFrame with zeros
            df_right_hand = pd.DataFrame(columns=[f'right_hand_{i}' for i in range(21 * 3)])

        df_combined = pd.concat([df_pose, df_left_hand, df_right_hand], axis=1)
        df_combined.to_csv(label + "_" + os.path.splitext(video_file)[0] + ".csv", index=False)

        cap.release()
        cv2.destroyAllWindows()

        processed_videos += 1

# Beispielaufrufe für verschiedene Ordner und Bewegungen
process_videos_in_folder("/home/prolab/Schreibtisch/Neue_Masterarbeit/Dataset/Data_sorted/combined_hinlangen_rh", "hinlangen_rh")
