import os
import pandas as pd

# Einlesen der CSV-Dateien
df1 = pd.read_csv("/home/prolab/Schreibtisch/Neue_Masterarbeit/Movement_Classification/Body_Movement_Classification/Test/predictions.csv")
df2 = pd.read_csv("/home/prolab/Schreibtisch/Neue_Masterarbeit/Movement_Classification/Object_Hand_Movement_Classification/object_coordinates.csv")

# Zusammenführen der Datenframes anhand der Spalte "Frame_Number"
merged_df = pd.merge(df1, df2, on='Frame_Number', how='inner')

# Speichern des zusammengeführten Dataframes als neue CSV-Datei
merged_df.to_csv("/home/prolab/Schreibtisch/Neue_Masterarbeit/Fusion/Fusion_Data/CSV_Combined/Merged_Data.csv", index=False)

# Löschen der alten CSV-Dateien
os.remove("/home/prolab/Schreibtisch/Neue_Masterarbeit/Movement_Classification/Body_Movement_Classification/Test/predictions.csv")
os.remove("/home/prolab/Schreibtisch/Neue_Masterarbeit/Movement_Classification/Object_Hand_Movement_Classification/object_coordinates.csv")
