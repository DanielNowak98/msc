import pandas as pd
import ast

# Lade die beiden CSV-Dateien in Pandas DataFrames
df1 = pd.read_csv("/home/prolab/Schreibtisch/Neue_Masterarbeit/MTM-Analysis/predictions.csv")
df2 = pd.read_csv("/home/prolab/Schreibtisch/Neue_Masterarbeit/Movement_Classification/Object_Hand_Movement_Classification/object_coordinates.csv")

# Funktion zum Extrahieren der relevanten Informationen aus der Zeichenfolge in Object_Info-Spalte
def extract_info(row):
    info_dict = ast.literal_eval(row)
    object_id = info_dict['Object_ID']
    status = info_dict['Status']
    class_ = info_dict['Class']
    return object_id, status, class_

# Extrahiere die Informationen und füge sie als neue Spalten hinzu
df2[['Object_ID', 'Status', 'Class']] = df2['Object_Info'].apply(lambda x: pd.Series(extract_info(x)))

# Entferne die nicht mehr benötigte Spalte "Object_Info"
df2.drop(columns=['Object_Info'], inplace=True)

# Füge die beiden DataFrames zusammen, indem du die Frame-Nummern als Schlüssel verwendest
merged_df = pd.merge(df1, df2, on="Frame_Number")

# Zeige das kombinierte DataFrame an
print(merged_df)

#merged_df.to_csv("TEST.csv")

# Finde den Index des ersten Auftretens von 'Grasped with right Hand'
index_grasped = merged_df.index[merged_df['Status'] == 'Grasped with right Hand'].min()

# Finde den Index des ersten Auftretens von 'AP_LH_ST'
index_ap_lh_st = merged_df.index[merged_df['Predictions'] == 'AP_RH_ST'].min()

# Starte von der niedrigsten Frame-Nummer und iteriere durch den DataFrame
for frame_number in range(merged_df['Frame_Number'].min(), merged_df['Frame_Number'].max() + 1):
    # Finde den Index des ersten Auftretens von 'Grasped with right Hand' nach dem aktuellen Frame
    index_grasped_after_frame = merged_df.index[(merged_df['Status'] == 'Grasped with right Hand') & (merged_df['Frame_Number'] > frame_number)].min()
    # Überprüfe, ob 'Grasped with right Hand' nach dem aktuellen Frame auftritt
    if pd.notnull(index_grasped_after_frame):
        # Berechne die Differenz der Frame-Nummern zwischen 'AP_LH_ST' und dem ersten Auftreten von 'Grasped with right Hand' nach dem aktuellen Frame
        difference_frame_numbers = merged_df.loc[index_grasped_after_frame, 'Frame_Number'] - merged_df.loc[index_ap_lh_st, 'Frame_Number']
        print("Für Frame", frame_number, "ist die Differenz der Frame-Nummern zwischen 'AP_RH_ST' und dem ersten Auftreten von 'Grasped with right Hand':", difference_frame_numbers)
        break  # Stoppe die Iteration, sobald die Differenz berechnet wurde

