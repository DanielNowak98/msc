#!/bin/bash

# Wechseln Sie zum Verzeichnis des zweiten Skripts
cd ~/Schreibtisch/Neue_Masterarbeit/Movement_Classification/Body_Movement_Classification/Test

# Führen Sie das zweite Python-Skript aus
python inference_lstm.py

# Wechseln Sie zum Verzeichnis des ersten Skripts
cd ~/Schreibtisch/Neue_Masterarbeit/Movement_Classification/Object_Hand_Movement_Classification

# Führen Sie das erste Python-Skript aus
python hand-object-interaction.py

