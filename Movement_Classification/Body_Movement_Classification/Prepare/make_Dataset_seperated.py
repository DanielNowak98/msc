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
                #print(f"Processing frame {frame_count} for video: {video_file}")

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

                cv2.imshow("image", frame)
                if cv2.waitKey(1) == ord('q'):  # Hier können Sie eine maximale Anzahl von Frames festlegen
                    break
            else:
                break

        df_pose = pd.DataFrame(lm_list_pose)
        df_left_hand = pd.DataFrame(lm_list_left_hand)
        df_right_hand = pd.DataFrame(lm_list_right_hand)

        df_pose.to_csv(label + "_pose_" + os.path.splitext(video_file)[0] + ".csv")
        df_left_hand.to_csv(label + "_left_hand_" + os.path.splitext(video_file)[0] + ".csv")
        df_right_hand.to_csv(label + "_right_hand_" + os.path.splitext(video_file)[0] + ".csv")

        cap.release()
        cv2.destroyAllWindows()

        processed_videos += 1
        #print(f"Processed {processed_videos}/{total_videos} videos.")

# Beispielaufrufe für verschiedene Ordner und Bewegungen
process_videos_in_folder("/home/prolab/Schreibtisch/Neue_Masterarbeit/Dataset/Data_sorted/combined_hinlangen_rh", "hinlangen_rh")
