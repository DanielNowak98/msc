import cv2
import mediapipe as mp
import pandas as pd
import os
from tqdm import tqdm

def process_videos_in_folder(folder_path, label):
    mpPose = mp.solutions.pose
    pose = mpPose.Pose()
    mpDraw = mp.solutions.drawing_utils

    lm_list = []

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
            cv2.circle(img, (cx, cy), 10, (0, 0, 255), cv2.FILLED)
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
                results = pose.process(frameRGB)

                if results.pose_landmarks:
                    lm = make_landmark_timestep(results)
                    lm_list.append(lm)
                    frame = draw_landmark_on_image(mpDraw, results, frame)

                cv2.imshow("image", frame)
                if cv2.waitKey(1) == ord('q'):  # Hier können Sie eine maximale Anzahl von Frames festlegen
                    break
            else:
                break

        df = pd.DataFrame(lm_list)
        df.to_csv(label + "_" + os.path.splitext(video_file)[0] + ".csv")

        cap.release()
        cv2.destroyAllWindows()

        processed_videos += 1
        #print(f"Processed {processed_videos}/{total_videos} videos.")

# Beispielaufrufe für verschiedene Ordner und Bewegungen
process_videos_in_folder("/home/prolab/Schreibtisch/Neue_Masterarbeit/Dataset/Data_sorted/combined_neutral_lh", "Neutral_lh")