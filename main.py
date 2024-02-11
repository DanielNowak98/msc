import subprocess

def execute_Detections():

    detection_script_path = "/home/prolab/Schreibtisch/Neue_Masterarbeit/run_detections.sh"

    # Befehl zum Ausführen des Shell-Skripts
    command = [detection_script_path]

    try:
        # Shell-Skript ausführen
        subprocess.run(command, check=True)
        print("Detection script executed successfully.")
    except subprocess.CalledProcessError as e:
        # Fehlerbehandlung, falls das Skript nicht erfolgreich ausgeführt wurde
        print(f"Error executing fusion script: {e}")


def execute_fusion_script():
    # Pfad zum Shell-Skript
    shell_script_path = "/home/prolab/Schreibtisch/Neue_Masterarbeit/combine_data.sh"

    # Befehl zum Ausführen des Shell-Skripts
    command = [shell_script_path]

    try:
        # Shell-Skript ausführen
        subprocess.run(command, check=True)
        print("Fusion script executed successfully.")
    except subprocess.CalledProcessError as e:
        # Fehlerbehandlung, falls das Skript nicht erfolgreich ausgeführt wurde
        print(f"Error executing fusion script: {e}")


def execute_analysis_script():
    # Pfad zum Shell-Skript
    shell_script_path = "/home/prolab/Schreibtisch/Neue_Masterarbeit/run_analysis.sh"

    # Befehl zum Ausführen des Shell-Skripts
    command = [shell_script_path]

    try:
        # Shell-Skript ausführen
        subprocess.run(command, check=True)
        print("Analysis script executed successfully.")
    except subprocess.CalledProcessError as e:
        # Fehlerbehandlung, falls das Skript nicht erfolgreich ausgeführt wurde
        print(f"Error executing Analysis script: {e}")

if __name__ == "__main__":
    print("#################################################################")
    print("Run Detections")
    print("#################################################################")
    execute_Detections()

    print("#################################################################")
    print("Combine CSV")
    print("#################################################################") 
    execute_fusion_script()

    print("#################################################################")
    print("Generate MTM-Analysis")
    print("#################################################################") 
    execute_analysis_script()

