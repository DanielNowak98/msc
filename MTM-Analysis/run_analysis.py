import pandas as pd
from tabulate import tabulate

df = pd.read_csv("/home/prolab/Schreibtisch/Neue_Masterarbeit/Fusion/Fusion_Data/CSV_Combined/Merged_Data.csv")

print(df.head())

# DataFrame nach dem Status "AP_RH_ST" durchsuchen
first_ap_rh_st_row = df[df['Predictions'] == 'AP_RH_ST'].iloc[0]

# Extrahiere den Frame_Number aus der ersten Zeile mit Status "AP_RH_ST"
frame_number_first_ap_rh_st = first_ap_rh_st_row['Frame_Number']

print("Frame_Number des ersten Auftretens von 'AP_RH_ST':", frame_number_first_ap_rh_st)

index_first_moving = None

# Initialisiere Variable, um die ID des bewegten Objekts zu speichern
moving_object_id = None

# Durchlaufe die Zeilen des DataFrame
for index, row in df.iterrows():
    # Überprüfe, ob der Status 'Moving' ist
    if row['Status'] == 'Moving':
        # Überprüfe, ob der vorherige Status 'Not Moving' war (oder es ist der erste Status im DataFrame)
        if index == 0 or df.loc[index - 1, 'Status'] == 'Not Moving':
            index_first_moving = index
            moving_object_id = row['Class']  # Speichere die Objekt-ID des bewegten Objekts
            break  # Beende die Schleife, sobald der erste Wechsel gefunden wurde

if index_first_moving is not None:
    frame_number_first_moving = df.loc[index_first_moving, 'Frame_Number']
    print("Frame_Number des ersten Wechsels von 'Not Moving' zu 'Moving':", frame_number_first_moving)
    print("Objekt-Klasse des bewegten Objekts:", moving_object_id)
else:
    print("Es wurde kein Wechsel von 'Not Moving' zu 'Moving' gefunden.")

# Überprüfe, ob ein Wechsel von 'Not Moving' zu 'Moving' gefunden wurde
if index_first_moving is not None:
    # Initialisiere Variablen, um den Index des erneuten Wechsels zu 'Not Moving' und den Zähler für 'Not Moving'-Frames zu speichern
    index_revert_not_moving = None
    not_moving_counter = 0

    # Durchlaufe die Zeilen des DataFrame, beginnend ab dem Index des ersten Wechsels zu 'Moving'
    for index, row in df.iloc[index_first_moving:].iterrows():
        # Überprüfe, ob der Status 'Not Moving' ist
        if row['Status'] == 'Not Moving':
            # Überprüfe, ob der 'Not Moving'-Status mindestens 10 Frame_Numbers anhält
            if not_moving_counter >= 10:
                # Überprüfe, ob der Status in der Spalte 'Predictions' gleichzeitig 'Neutral' ist
                if all(df.loc[index - not_moving_counter:index, 'Predictions'] == 'Neutral'):
                    index_revert_not_moving = index - not_moving_counter
                    break  # Beende die Schleife, sobald die Bedingung erfüllt ist
            else:
                not_moving_counter += 1
        else:
            not_moving_counter = 0  # Setze den Zähler zurück, wenn der Status nicht 'Not Moving' ist

    if index_revert_not_moving is not None:
        frame_number_revert_not_moving = df.loc[index_revert_not_moving, 'Frame_Number']
        print("Frame_Number, ab dem der Status sich wieder auf 'Not Moving' ändert und für mindestens 10 Frame_Numbers anhält:", frame_number_revert_not_moving)
        
        # Berechne die Anzahl der Frame_Numbers zwischen den beiden Frame_Numbers
        frame_numbers_difference = frame_number_revert_not_moving - frame_number_first_ap_rh_st
        print("Anzahl der Frame_Numbers zwischen dem ersten Auftreten von 'AP_RH_ST' und dem erneuten Wechsel zu 'Not Moving':", frame_numbers_difference)
    else:
        print("Es wurde kein Wechsel zu 'Not Moving' gefunden, der für mindestens 10 Frame_Numbers anhält.")
else:
    print("Es wurde kein Wechsel von 'Not Moving' zu 'Moving' gefunden.")

# Berechnung der Bewegungslängen in X- und Y-Richtung
start_x, start_y = df.iloc[index_first_moving]['Center_X'], df.iloc[index_first_moving]['Center_Y']
end_x, end_y = df.iloc[index_revert_not_moving]['Center_X'], df.iloc[index_revert_not_moving]['Center_Y']
movement_length_x = abs(end_x - start_x)
movement_length_y = abs(end_y - start_y)

print("##########################")
print("SUMMARY")
print("##########################")
print("-> Art der Bewegung:", first_ap_rh_st_row['Predictions'])
print("-> Bewegung [Person] für", frame_numbers_difference, "Frames")
print("-> Bewegtes Objekt:", moving_object_id)
print("-> Bewegung [Objekt] für Distanz:", frame_numbers_difference, "Frames")
print("-> Bewegungslänge in X-Richtung:", movement_length_x)
print("-> Bewegungslänge in Y-Richtung:", movement_length_y)
print("----------------------------------------------------------------------------------------------------")

knowledge_df = pd.read_csv("/home/prolab/Schreibtisch/Neue_Masterarbeit/MTM-Analysis/object.csv")  # Pfade entsprechend anpassen

# Annahme: moving_object_id ist die Objekt-ID des bewegten Objekts
moving_object_id = "bottle"  # Beispielwert

# Suche nach Informationen zum bewegten Objekt in der Wissensdatenbank
object_info = knowledge_df[knowledge_df['object'] == moving_object_id]

def convert_Frame_Number_to_TMU_and_Seconds(inp):
    #1 TMU = 0,036 Sekunden
    #1 Sekunde = 50 Frames
    #1 Frame = TMU?
    TMU = inp*0.5556 
    return TMU/2

# Überprüfe, ob Informationen zum bewegten Objekt gefunden wurden
if not object_info.empty:
    # Extrahiere die Informationen aus der Wissensdatenbank
    usabel_as_tool = object_info['usabel_as_tool'].iloc[0]
    weight = object_info['weight'].iloc[0]
    case = object_info['case'].iloc[0]
    print("----------------------------------------------------------------------------------------------------")   
    # Ausgabe der Vorbereitung für die MTM-Analyse
    print("##########################")
    print("PREPARE MTM ANALYSIS")
    print("##########################")
    print("----------------------------------------------------------------------------------------------------")
    print("Video FPS:", 50)
    print("----------------------------------------------------------------------------------------------------")
    print("-> Bewegtes Objekt:", moving_object_id)
    print("----> usabel_as_tool:", usabel_as_tool)
    print("----> weight:", weight)
    print("----> Handhabbarkeit:", case)
    print("----> Bewegungslänge in X-Richtung [Pixel]:", movement_length_x)
    print("----> Bewegungslänge in Y-Richtung [Pixel]:", movement_length_y)
    print("----> Bewegungslänge in X-Richtung [cm]:", movement_length_x/10)
    print("----> Bewegungslänge in Y-Richtung [cm]:", movement_length_y/10)

    print("-> Art der Bewegung:", first_ap_rh_st_row['Predictions'])
    print("----> Bewegungslänge [Frames]:", frame_numbers_difference)
    print("----> Bewegungslänge [TMU]:", convert_Frame_Number_to_TMU_and_Seconds(frame_numbers_difference))
    print("----------------------------------------------------------------------------------------------------")

else:
    print("Keine Informationen für das bewegte Objekt in der Wissensdatenbank gefunden.")


length = movement_length_x/10

print("length:", length)
print("case", case)
print("weight", weight)

csv_file_MTM = '/home/prolab/Schreibtisch/Neue_Masterarbeit/MTM-Analysis/mtm_UAS-Datenkarte.csv'

import csv

# Funktion zum Finden des passenden Codes
def find_code_and_tmu(length, weight, case, csv_file):
    with open(csv_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if (row['Length'] == '<20' and length < 20) or (row['Length'] == '>20' and length >= 20):
                if row['Case'] == case and float(row['weight'].replace('<', '').replace('>', '')) >= weight:
                    return row['Code'], row['TMU']
        return None, None
    
code, tmu = find_code_and_tmu(length, weight, case, csv_file_MTM)
if code:
    print(f"Der passende Code ist: {code}")
else:
    print("Kein passender Code gefunden.")

print("----------------------------------------------------------------------------------------------------")   
# Ausgabe der Vorbereitung für die MTM-Analyse
print("##########################")
print("MTM ANALYSIS")
print("##########################")

if first_ap_rh_st_row['Predictions']:
    do = "Aufnehmen und Platzieren, stehend"

else:
    do = "UNBEKANNTE BEWEGUNG"


descript = f"{do} des Objektes {moving_object_id} in das Sichtfeld"


# Erstellen des DataFrames
df = pd.DataFrame({
    'Beschreibung der Tätigkeit': [descript],
    'Code': code,
    'TMU': tmu,
    'Anzahl x Häufigkeit': 1
})
# DataFrame als Tabelle formatieren
table = tabulate(df, headers='keys', tablefmt='grid', showindex=False)

# Tabelle in eine PDF-Datei schreiben
with open('/home/prolab/Schreibtisch/Neue_Masterarbeit/MTM-ANALYSIS.pdf', 'w') as f:
    f.write(table)

print("----> MTM ANALYIS Saved to MTM-ANALYSIS.pdf <----")