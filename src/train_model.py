import os
import numpy as np
import pickle
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import face_recognition

# Step 1: Prepare Dataset
def get_face_encodings(image_path):
    try:
        image = face_recognition.load_image_file(image_path)
        face_encodings = face_recognition.face_encodings(image)
        if face_encodings:
            return face_encodings[0]
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
    return None

def prepare_dataset(dataset_path):
    labels = []
    encodings = []
    for person_name in os.listdir(dataset_path):
        person_folder = os.path.join(dataset_path, person_name)
        if os.path.isdir(person_folder):
            for image_name in os.listdir(person_folder):
                image_path = os.path.join(person_folder, image_name)
                print(f"Processing {image_path}...")
                encoding = get_face_encodings(image_path)
                if encoding is not None:
                    encodings.append(encoding)
                    labels.append(person_name)
                else:
                    print(f"Warning: No encoding found for image: {image_path}")
    print(f"Found {len(np.unique(labels))} unique classes.")
    return np.array(encodings), np.array(labels)

# Step 2: Train Classifier
def train_classifier(encodings, labels):
    X_train, X_test, y_train, y_test = train_test_split(
        encodings, labels, test_size=0.2, random_state=42, stratify=labels
    )

    print(f"Training set classes: {np.unique(y_train, return_counts=True)}")
    print(f"Testing set classes: {np.unique(y_test, return_counts=True)}")

    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        print("Error: One of the datasets has less than 2 classes. Add more data.")
        return None

    clf = svm.SVC(gamma='scale')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
    return clf

if __name__ == "__main__":
    dataset_path = r'D:\python\MachineLearning\face-recognition-attendance-system\data'

    # ✅ Define models directory
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Define models directory
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    os.makedirs(MODELS_DIR, exist_ok=True)

    encodings, labels = prepare_dataset(dataset_path)

    print(f"Number of encodings: {len(encodings)}")
    print(f"Encodings shape: {encodings.shape if encodings.size else 'Empty'}")
    print(f"Number of labels: {len(labels)}")
    print(f"Unique classes: {np.unique(labels)}")

    if encodings.size == 0 or len(labels) == 0:
        print("Error: No encodings or labels found. Check your dataset path and contents.")
        exit()

    if len(np.unique(labels)) < 2:
        print("Error: Less than 2 unique classes found. Add more data from different classes.")
        exit()

    # ✅ Save encodings to models folder
    with open(os.path.join(MODELS_DIR, 'encodings.pkl'), 'wb') as f:
        pickle.dump((encodings, labels), f)

    clf = train_classifier(encodings, labels)
    if clf is None:
        print("Classifier training failed due to insufficient data. Exiting.")
        exit()

    # ✅ Save classifier to models folder
    with open(os.path.join(MODELS_DIR, 'classifier.pkl'), 'wb') as f:
        pickle.dump(clf, f)

    print(f"Training completed and model saved in: {MODELS_DIR}")
