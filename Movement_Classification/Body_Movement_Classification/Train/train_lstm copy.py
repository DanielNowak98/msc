import numpy as np
import pandas as pd
from keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import os
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from keras.optimizers import Adam


def read_data_from_folders(folder_paths, label):
    X = []
    y = []
    no_of_timesteps = 2

    for folder_path in folder_paths:
        for txt_file in os.listdir(folder_path):
            if txt_file.endswith(".csv"):
                txt_path = os.path.join(folder_path, txt_file)
                df = pd.read_csv(txt_path)
                dataset = df.iloc[:, 1:].values
                n_sample = len(dataset)

                for i in range(no_of_timesteps, n_sample):
                    X.append(dataset[i - no_of_timesteps:i, :])
                    y.append(label)

    X, y = np.array(X), np.array(y)
    return X, y

# Define the folder paths and corresponding labels
folder_paths = [
    "/home/prolab/Schreibtisch/Neue_Masterarbeit/Dataset/Data_CSV/hinlangen_rh",
    "/home/prolab/Schreibtisch/Neue_Masterarbeit/Dataset/Data_CSV/hinlangen_lh"
]

# Corresponding labels for each folder
labels = [0, 1]

# Read data from folders
for i, folder_path in enumerate(folder_paths):
    X_temp, y_temp = read_data_from_folders([folder_path], labels[i])
    if i == 0:
        X, y = X_temp, y_temp
    else:
        X = np.concatenate((X, X_temp), axis=0)
        y = np.concatenate((y, y_temp), axis=0)

print(X.shape, y.shape)

# Convert labels to one-hot encoding
y_one_hot = to_categorical(y, num_classes=len(labels))

X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2)

# Define the model
model = Sequential()
model.add(Bidirectional(LSTM(units=100, return_sequences=True), input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.5))
model.add(Bidirectional(LSTM(units=100, return_sequences=True)))
model.add(Dropout(0.5))
model.add(Bidirectional(LSTM(units=100, return_sequences=True)))
model.add(Dropout(0.5))
model.add(Bidirectional(LSTM(units=100, return_sequences=True)))
model.add(Dropout(0.5))
model.add(Bidirectional(LSTM(units=100)))
model.add(Dropout(0.5))
model.add(Dense(units=64, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(units=len(labels), activation="softmax"))

optimizer = Adam(lr=0.001)
model.compile(optimizer=optimizer, metrics=['accuracy'], loss="categorical_crossentropy")

# Print the model summary
model.summary()

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_test, y_test))

model.save("model.h5")


# Predict on test set
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Generate confusion matrix
conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
