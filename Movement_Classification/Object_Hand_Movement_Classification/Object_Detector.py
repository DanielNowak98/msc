import cv2
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
        header = ['Frame_Number', 'Object_Info']
        self.csv_writer.writerow(header)

        self.prev_frame_objects = None  # List to store detected objects in the previous frame

    def detect_objects(self, frame, frame_number):
        detect_params = self.model.predict(source=[frame], conf=self.conf_threshold, save=False)
        return detect_params

    def draw_objects(self, frame, detect_params, frame_number):
        frame_objects = []  # List to store detected objects in the current frame

        for i, box in enumerate(detect_params[0].boxes):
            clsID = box.cls[0].item()
            conf = box.conf[0].item()
            bb = box.xyxy[0].cpu().numpy()

            center_x = int((bb[0] + bb[2]) / 2)
            center_y = int((bb[1] + bb[3]) / 2)

            # Assume no movement initially
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

            font = cv2.FONT_HERSHEY_SIMPLEX
            text = f"Object {i + 1}: {self.class_list[int(clsID)]} {round(conf * 100, 2)}% - {status}"
            text_position = (10, 50 + i * 30)

            # Ensure text is not overlaid on previous text
            while text_position in [prev_text[1] for prev_text in frame_objects]:
                text_position = (text_position[0], text_position[1] + 30)

            cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)

        # Check object movement by comparing with the previous frame
        if self.prev_frame_objects is not None:
            for prev_obj, curr_obj in zip(self.prev_frame_objects, frame_objects):
                if prev_obj[2:] != curr_obj[2:]:  # Compare coordinates
                    movement_text = f"Object {curr_obj[0]} moved in frame {frame_number}"
                    print(movement_text)

                    # Check movement distance
                    movement_distance = ((curr_obj[2] - prev_obj[2])**2 + (curr_obj[3] - prev_obj[3])**2)**0.5
                    if movement_distance > self.movement_threshold:
                        # Update the status in the current frame_objects
                        curr_obj[-1] = "Moving"

                        # Draw movement information on the frame
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        # cv2.putText(frame, movement_text, (10, 50), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Draw all object information on the frame
        for i, obj_info in enumerate(frame_objects):
            text = f"Object {obj_info[0]}: {obj_info[1]} {round(conf * 100, 2)}% - {obj_info[-1]}"
            text_position = (10, 50 + i * 30)
            cv2.putText(frame, text, text_position, font, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Write one row per frame with all detected objects and their status
        row = [frame_number]
        for obj_info in frame_objects:
            row.append({
                'Object_ID': obj_info[0],
                'Class': obj_info[1],
                'Center_X': obj_info[2],
                'Center_Y': obj_info[3],
                'Status': obj_info[-1]
            })

        self.csv_writer.writerow(row)

        # Save current frame_objects as previous for the next iteration
        self.prev_frame_objects = frame_objects

    def release_resources(self):
        self.csv_file.close()


def run():
    model_path = "weights/yolov8x.pt"
    class_list_path = "utils/coco.txt"
    conf_threshold = 0.45
    movement_threshold = 5

    detector = ObjectDetector(model_path, class_list_path, conf_threshold, movement_threshold)

    cap = cv2.VideoCapture("/home/prolab/Schreibtisch/Neue_Masterarbeit/Dataset/HAVID_RGB/assembly_dataset_mp4_blurred/s01/S01A01I01M0.mp4")  # Pfad zu Ihrem Video
    if not cap.isOpened():
        print("Error: Unable to open video.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        #print("Frame Number:", frame_number)
        detect_params = detector.detect_objects(frame, frame_number)
        detector.draw_objects(frame, detect_params, frame_number)

        cv2.imshow("Object Detection", frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    detector.release_resources()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run()
