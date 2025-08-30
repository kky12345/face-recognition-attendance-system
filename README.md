# Face Recognition Attendance System

A machine learning project that uses **face recognition** to automatically mark attendance.  
The system captures images, extracts facial encodings, trains a classifier, and records attendance in a CSV file.

---

## 📂 Project Structure

```

face-recognition-attendance-system/
│── data/                          # raw images, organized by person
│   ├── person1/
│   ├── person2/
│   └── ...
│
│── models/                        # trained models, encodings
│   ├── encodings.pkl
│   └── classifier.pkl
│
│── notebook/
│   └── face\_recognition\_demo.ipynb
│
│── src/
│   ├── train\_model.py
│   ├── test\_model.py
│   └── utils.py (optional helpers)
│
│── outputs/
│   └── attendance.csv
│
│── requirements.txt
│── README.md
│── .gitignore

````

---

## 🚀 Features

- Add new people by placing their images in `data/person_name/`.
- Extract facial encodings and train a classifier.
- Recognize faces in real-time or from images.
- Record recognized faces into `outputs/attendance.csv`.

---

## 💻 Installation

1. Clone the repository:

```bash
git clone <your-repo-url>
cd face-recognition-attendance-system
````

2. Create a virtual environment (recommended):

```bash
python -m venv venv
# Activate the environment
# On Linux/Mac
source venv/bin/activate
# On Windows
venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🖥️ Usage

### 1. Prepare the dataset

Place images inside the `data/` folder, organized by person:

```
data/
├── alice/
│   ├── img1.jpg
│   ├── img2.jpg
├── bob/
│   ├── img1.jpg
│   ├── img2.jpg
```

### 2. Train the model

```bash
python src/train_model.py
```

This will generate:

* `models/encodings.pkl`
* `models/classifier.pkl`

### 3. Test recognition

```bash
python src/test_model.py
```

Recognized names will be appended to:

```
outputs/attendance.csv
```

### 4. Explore with Jupyter Notebook

A demo notebook is available at:

```
notebook/face_recognition_demo.ipynb
```

> **Note:** The notebook assumes it lives inside `notebook/` and resolves paths relative to the project root (`..`).

---

## 📊 Example Output

`outputs/attendance.csv`

```
Name,Date,Time
Elon Musk,2025-08-30,16:35:40
Mark Zucker Burg,2025-08-30,16:36:08
Jeff Bezos,2025-08-30,16:36:11
```

---

## 🛠️ Tech Stack

* Python
* OpenCV
* dlib / face\_recognition
* scikit-learn
* Jupyter Notebook

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you’d like to change.

