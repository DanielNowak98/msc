import cv2
import mediapipe as mp
from ultralytics import YOLO
import csv
import time
import random

class ObjectDetector:
    def __init__(self, model_path, class_list_path, conf_threshold=0.45, movement_threshold=10):
        self.model = YOLO(model_path, "v8")
        self.conf_threshold = conf_threshold
        self.movement_threshold = movement_threshold

        with open(class_list_path, "r") as file:
            self.class_list = file.read().split("\n")

        self.detection_colors = [
            (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            for _ in range(len(self.class_list))
        ]

        self.csv_file = open('object_coordinates.csv', 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        header = ['Frame_Number', 'Object_ID', 'Class', 'Center_X', 'Center_Y', 'Status' , 'Keypoint_15_X', 'Keypoint_15_Y', 'Keypoint_16_X', 'Keypoint_16_Y', 'LH-Grasp_Status', 'RH-Grasp_Status']
        self.csv_writer.writerow(header)

        self.prev_frame_objects = None
        self.current_frame_data = []

        # Initialize Mediapipe Pose
        self.mp_pose = mp.solutions.pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.pose_draw = mp.solutions.drawing_utils

    def detect_objects(self, frame, frame_number):
        detect_params = self.model.predict(source=[frame], conf=self.conf_threshold, save=False)

        frame_objects = []

        for i, box in enumerate(detect_params[0].boxes):
            clsID = box.cls[0].item()
            conf = box.conf[0].item()

            if self.class_list[int(clsID)] == "bottle" and conf >= self.conf_threshold:
                bb = box.xyxy[0].cpu().numpy()

                center_x = int((bb[0] + bb[2]) / 2)
                center_y = int((bb[1] + bb[3]) / 2)

                status = "Not Moving"

                object_info = [i + 1, self.class_list[int(clsID)], center_x, center_y, status]
                frame_objects.append(object_info)

                cv2.rectangle(
                    frame,
                    (int(bb[0]), int(bb[1])),
                    (int(bb[2]), int(bb[3])),
                    self.detection_colors[int(clsID)],
                    3,
                )

                cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)

        if self.prev_frame_objects is not None:
            for prev_obj, curr_obj in zip(self.prev_frame_objects, frame_objects):
                if prev_obj[2:] != curr_obj[2:]:
                    movement_distance = (curr_obj[2] - prev_obj[2])**2 + (curr_obj[3] - prev_obj[3])**2
                    if movement_distance > self.movement_threshold ** 2:
                        curr_obj[-1] = "Moving"
                        continue

        self.current_frame_data = [frame_number]
        for obj_info in frame_objects:
            self.current_frame_data.append({
                'Object_ID': obj_info[0],
                'Class': obj_info[1],
                'Center_X': obj_info[2],
                'Center_Y': obj_info[3],
                'Status': obj_info[-1],
            })

        self.prev_frame_objects = frame_objects

    def draw_objects(self, frame, detect_params, frame_number):
        for obj_info in self.prev_frame_objects:
            obj_x, obj_y = obj_info[2], obj_info[3]
            cv2.putText(frame, f'Object {obj_info[0]}: ({obj_x}, {obj_y})', (obj_x, obj_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

    def detect_pose(self, frame, frame_number):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.mp_pose.process(frame_rgb)
        if results.pose_landmarks:
            # Draw Mediapipe Pose landmarks
            self.pose_draw.draw_landmarks(frame, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
            # Draw and label Keypoints 15 and 16
            keypoints = results.pose_landmarks.landmark
            if keypoints:
                keypoint_15 = keypoints[15]
                keypoint_16 = keypoints[16]
                height, width, _ = frame.shape
                keypoint_15_x, keypoint_15_y = int(keypoint_15.x * width), int(keypoint_15.y * height)
                keypoint_16_x, keypoint_16_y = int(keypoint_16.x * width), int(keypoint_16.y * height)
                cv2.circle(frame, (keypoint_15_x, keypoint_15_y), 5, (255, 0, 0), -1)
                cv2.circle(frame, (keypoint_16_x, keypoint_16_y), 5, (255, 0, 0), -1)
                # Write text labels
                cv2.putText(frame, f'Keypoint 15, X: {keypoint_15_x}, Y: {keypoint_15_y}', (keypoint_15_x, keypoint_15_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, f'Keypoint 16, X: {keypoint_16_x}, Y: {keypoint_16_y}', (keypoint_16_x, keypoint_16_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                
            for obj_info in self.prev_frame_objects:
                obj_x, obj_y = obj_info[2], obj_info[3]
                obj_status = obj_info[-1]
                # Check if object is near left hand
                object_threshold = 20  # Adjust as needed
                left_hand_grasp = ""
                if abs(obj_x - keypoint_15_x) < object_threshold or abs(obj_y - keypoint_15_y) < object_threshold:
                    if obj_status == "Moving":
                        cv2.putText(frame, f'Object {obj_info[1]} grasped with left Hand', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                        left_hand_grasp = f'Object {obj_info[1]} grasped with left Hand'

                # Check if object is near right hand
                right_hand_grasp = ""
                if abs(obj_x - keypoint_16_x) < object_threshold or abs(obj_y - keypoint_16_y) < object_threshold:
                    if obj_status == "Moving":
                        cv2.putText(frame, f'Object {obj_info[1]} grasped with right Hand', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                        right_hand_grasp = f'Object {obj_info[1]} grasped with right Hand'
                        
                # Write to CSV
                row = [frame_number]
                row.extend(obj_info[0:4])  # Add object ID, class, center X, center Y
                row.append(obj_info[-1])  # Add status
                row.extend([keypoint_15_x, keypoint_15_y, keypoint_16_x, keypoint_16_y])  # Add keypoint coordinates
                row.append(left_hand_grasp)  # Add left hand grasp information
                row.append(right_hand_grasp)  # Add right hand grasp information
                self.csv_writer.writerow(row)




    def release_resources(self):
        self.csv_file.close()

def run():
    MODEL_PATH = "weights/yolov8x.pt"
    CLASS_LIST_PATH = "utils/coco.txt"
    CONF_THRESHOLD = 0.45
    MOVEMENT_THRESHOLD = 5

    detector = ObjectDetector(MODEL_PATH, CLASS_LIST_PATH, CONF_THRESHOLD, MOVEMENT_THRESHOLD)

    cap = cv2.VideoCapture("/home/prolab/Schreibtisch/Neue_Masterarbeit/Dataset/Videos_own_Dataset/C0028.MP4")
    if not cap.isOpened():
        print("Error: Unable to open video.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        detector.detect_objects(frame, frame_number)
        detector.draw_objects(frame, None, frame_number)
        detector.detect_pose(frame, frame_number)  # Detect and draw pose

        cv2.imshow("Object Detection", frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    detector.release_resources()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run()
