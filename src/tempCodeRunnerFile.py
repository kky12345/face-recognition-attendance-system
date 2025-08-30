import cv2
import numpy as np
import pickle
import face_recognition
import csv
from datetime import datetime
import os

# === Define paths ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # main project folder
print(f"Base Directory: {BASE_DIR}")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)  # make sure output folder exists

ATTENDANCE_FILE = os.path.join(OUTPUT_DIR, "attendance.csv")

# Function to mark attendance in CSV
def mark_attendance(name, filename=ATTENDANCE_FILE):
    now = datetime.now()
    dt_string = now.strftime('%Y-%m-%d')   # Date
    tm_string = now.strftime('%H:%M:%S')   # Time

    file_exists = os.path.isfile(filename)

    # If file doesn't exist, create it with header
    if not file_exists:
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Name", "Date", "Time"])

    # Check if this name already has attendance for today
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header if exists
        for row in reader:
            if len(row) >= 2 and row[0] == name and row[1] == dt_string:
                print(f"{name} already marked for {dt_string}")
                return  # Don't add again

    # If not marked, add new record
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([name, dt_string, tm_string])
        print(f"Attendance marked for {name} at {dt_string} {tm_string}")

# Step 3: Recognize Faces
def recognize_faces(frame, clf, known_face_names, known_face_encodings, recognized_names, distance_threshold=0.6):
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)
    names = []

    for face_encoding in face_encodings:
        distances = np.linalg.norm(known_face_encodings - face_encoding, axis=1)
        min_distance = np.min(distances)
        if min_distance < distance_threshold:
            best_match_index = np.argmin(distances)
            name = known_face_names[best_match_index]
            names.append(name)
            if name not in recognized_names:
                recognized_names.add(name)
                mark_attendance(name)  # Log attendance once per day
                print(f"Recognized: {name} with distance: {min_distance}")
        else:
            names.append("Unknown")
            print("Face not recognized.")

    # Remove names that are not currently in frame
    removed_names = recognized_names.copy()
    for name in removed_names:
        if name not in names:
            recognized_names.remove(name)

    return face_locations, names

# Step 4: Run the Face Recognition System
def run_recognition_system():
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    CLASSIFIER_PATH = os.path.join(MODELS_DIR, "classifier.pkl")
    ENCODINGS_PATH = os.path.join(MODELS_DIR, "encodings.pkl")

    with open(CLASSIFIER_PATH, 'rb') as f:
        clf = pickle.load(f)

    with open(ENCODINGS_PATH, 'rb') as f:
        known_face_encodings, known_face_names = pickle.load(f)

    recognized_names = set()

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image from camera.")
            break

        face_locations, names = recognize_faces(frame, clf, known_face_names, known_face_encodings, recognized_names)

        for (top, right, bottom, left), name in zip(face_locations, names):
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_recognition_system()
