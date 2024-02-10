import csv

# 15 FPS 
# 1 TMU = 0,36 Sekunden
# 1 Sekunde = 15 Frames
# Also entspricht 1 Frame etwa 0,0667 Sekunden.

def summarize_csv(input_file):
    with open(input_file, 'r') as csv_file:
        reader = csv.reader(csv_file)
        next(reader)  # Überspringt die Header-Zeile

        current_prediction = None
        frame_count = 0

        for row in reader:
            frame, prediction = row

            if current_prediction == prediction:
                frame_count += 1
            else:
                if current_prediction is not None:
                    print(f'Prediction: {current_prediction} for {frame_count*2*0.06} Sekunden -> {((frame_count*2*0.06))/0.36} TMU')

                current_prediction = prediction
                frame_count = 1


summarize_csv('predictions.csv')
