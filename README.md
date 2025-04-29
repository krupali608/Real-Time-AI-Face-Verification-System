# Real-Time AI Face Verification System

This project implements a Real-Time AI Face Verification System that combines YOLOv8-based anti-spoofing detection and CNN-based gesture recognition (blink/smile) to provide secure two-factor face authentication. It is structured into three main modules: anti-spoofing (YOLOv8), gesture verification (CNN), and the integrated Flask-based user interface.

---

## Project Structure
```
/AntiSpoofing/
├── dataCollection.py         # Capture real/fake spoof dataset
├── splitData.py              # Split dataset into train/val/test
├── train.py                  # Train YOLOv8 on spoof dataset
├── main.py                   # Test YOLOv8 real/fake detection
├── yoloTest.py               # General YOLO object detection demo
├── faceDetectorTest.py       # Test face detection module (cvzone)

/GestureRecogntion/
├── train_gesture_model.py    # Prepare and train gesture classifier
├── cnn_gesture_model.py      # CNN architecture (MobileNetV2)
├── predict_gesture_model.py  # Real-time gesture prediction demo

/UI/
├── app.py                    # Main Flask App (UI integration)
├── index.html                # Frontend Web UI

/Models/
├── gesture_classifier_model.h5         # Trained CNN model
├── n_version_1.pt                       # YOLOv8 model
├── shape_predictor_68_face_landmarks.dat # Dlib facial landmark model
├── yolov8n.pt                           # YOLO model architecture

/EDA/
├── Gesture_EDA/                         # Visuals and metrics for gesture classification
├── Spoof_EDA/

/gesture_data/
├── blink/      # Raw blink gesture images
├── smile/      # Raw smile gesture images

/Project-Report/
├── CAPSTONE PROJECT REPORT-compressed.pdf
├── Literature-Review.pdf
├── POS-KrupaliShinde.pdf
├── realtime_ai_faceverification_ppt.pptx
```

---

## Dataset Structure and Requirements

### 1. Spoof Detection Dataset (YOLOv8)
- Real faces captured using a webcam.
- Fake faces printed on paper or shown on digital devices.
- Labelled in YOLO format.
- After capturing, real and fake samples must be organized separately.

```
/Dataset/Real/
/Dataset/Fake/
```

During data collection, all images and labels are initially saved in:
```
/Dataset/all/
├── image1.jpg
├── image1.txt
...
```

### 2. Gesture Recognition Dataset (CNN)
- Classes: `blink`, `smile`

```
/g_data/
├── blink/
├── smile/
```

> Total images used: approximately 17,000 across both datasets.

---

## Technologies Used
- Python 3.8+
- YOLOv8 (Ultralytics)
- TensorFlow / Keras
- OpenCV
- Flask
- Dlib
- cvzone

---

## Project Execution Workflow

### Anti-Spoofing (YOLOv8)
1. **Capture Dataset**
   ```bash
   python AntiSpoofing/dataCollection.py
   ```
2. **Split Dataset**
   ```bash
   python AntiSpoofing/splitData.py
   ```
3. **Train Spoof Detection Model**
   ```bash
   python AntiSpoofing/train.py
   ```
4. **Test Detection (Optional)**
   ```bash
   python AntiSpoofing/main.py
   ```

### Gesture Recognition (CNN)
1. **Prepare Gesture Images** manually or from webcam
2. **Train CNN Gesture Model**
   ```bash
   python GestureRecogntion/train_gesture_model.py
   ```
3. **Test Gesture Classifier (Optional)**
   ```bash
   python GestureRecogntion/predict_gesture_model.py
   ```

### Integrated Web Application (Flask UI)
Run the Flask app to combine both detection modules:
```bash
python UI/app.py
```

This will:
- Launch a webcam stream
- Use YOLOv8 to classify the face as real or fake
- If real, prompt for blink or smile
- Use CNN to verify the gesture

---

## Evaluation Metrics

### YOLOv8 Anti-Spoofing
- Precision: 98.9%
- Recall: 98.1%
- mAP@0.5: 99.2%

### CNN Gesture Recognition
- Accuracy: 100% on validation set
- F1-Score: 1.00 (blink and smile)

---

## Project Presentation

All final documentation and deliverables are included in the [Project-Report](./Project-Report/) folder.

Download directly:
- [Capstone Presentation (PPTX)](./Project-Report/realtime_ai_faceverification_ppt.pptx)
- [Capstone Report (PDF)](./Project-Report/CAPSTONE%20PROJECT%20REPORT-compressed.pdf)

---

## Notes
- Place all trained models in the `/Models/` folder:
  - `gesture_classifier_model.h5`
  - `n_version_1.pt`
  - `shape_predictor_68_face_landmarks.dat`
- A webcam is required to test the real-time app.
- Flask server runs locally at: `http://127.0.0.1:5000/`

---

## Future Enhancements
- Add more advanced gesture types (e.g., head tilt, nod, eye roll)
- Support multi-modal biometric verification (voice + face)
- Deploy the application on a cloud platform (e.g., Heroku, AWS, GCP)

---

Developed by: Krupali Shinde  
CS 668 - Analytics Capstone Project
