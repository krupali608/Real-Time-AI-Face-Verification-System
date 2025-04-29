# Real-Time AI Face Verification System

This project implements a Real-Time AI Face Verification System that combines YOLOv8-based anti-spoofing detection and CNN-based gesture recognition (blink/smile) to provide secure two-factor face authentication. It is structured into three main modules: anti-spoofing (YOLOv8), gesture verification (CNN), and the integrated Flask-based user interface.

---

## Project Structure
```
├── app.py                    # Main Flask App (UI integration)
├── index.html                # Frontend Web UI

# Anti-Spoofing (YOLOv8)
├── dataCollection.py         # Capture real/fake spoof dataset
├── splitData.py              # Split dataset into train/val/test
├── train.py                  # Train YOLOv8 on spoof dataset
├── main.py                   # Test YOLOv8 real/fake detection
├── yoloTest.py               # General YOLO object detection demo

# Gesture Classification (CNN)
├── train_gesture_model.py    # Prepare and train gesture classifier
├── cnn_gesture_model.py      # CNN architecture (MobileNetV2)
├── predict_gesture_model.py  # Real-time gesture prediction demo

# Utilities
├── faceDetectorTest.py       # Test face detection module (cvzone)
├── /models/                  # Folder for YOLO .pt and CNN .h5 models
├── /runs/                    # YOLOv8 training results, loss plots, metrics
```

---

## Dataset Requirements

### 1. Spoof Detection Dataset (YOLOv8)
- Real faces captured using a webcam.
- Fake faces printed on paper or shown on digital devices.
- Labelled in YOLO format.
- The Folder structure needs to be such that, when you are capturing Real Faces and after it is saved in /Dataset/all/, then it needs to be moved to /Dataset/Real
- Same goes for Fake, when you are capturing Fake Faces, and, after it is saved in /Dataset/all/, then it needs to be moved to /Dataset/Fake

Folder structure:
```
/Dataset/all/
├── image1.jpg
├── image1.txt
├── image2.jpg
├── image2.txt
...
```

### 2. Gesture Recognition Dataset (CNN)
- Two classes: `blink` and `smile`

Folder structure:
```
/g_data/
├── blink/
├── smile/
```

Total images used: approximately 17,000 across both datasets.

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
   python dataCollection.py
   ```
2. **Split Dataset**
   ```bash
   python splitData.py
   ```
3. **Train Spoof Detection Model**
   ```bash
   python train.py
   ```
4. **Test Detection (Optional)**
   ```bash
   python main.py
   ```

### Gesture Recognition (CNN)
1. **Collect Gesture Images** (manually or from webcam frame crops)
2. **Train CNN Gesture Model**
   ```bash
   python train_gesture_model.py
   ```
3. **Test Gesture Classifier (Optional)**
   ```bash
   python predict_gesture_model.py
   ```

### User Interface (Full System)
Run the integrated Flask application:
```bash
python app.py
```

This will:
- Open webcam stream
- Run YOLOv8 model to detect real/fake
- If real, prompt a gesture (blink/smile)
- Run CNN model to verify the gesture

---

## Evaluation Metrics

### YOLOv8 Anti-Spoofing
- Precision: 98.9%
- Recall: 98.1%
- mAP@0.5: 99.2%

### CNN Gesture Recognition
- Accuracy: 100% on validation set
- F1-Score: 1.00 for both blink and smile

## Project Presentation

Download the PowerPoint here:
[realtime_ai_faceverification_ppt.pptx](./realtime_ai_faceverification_ppt.pptx)
---

## Notes
- Ensure models are saved in the `/models/` directory:
  - `gesture_classifier_model.h5`
  - `n_version_1.pt` (YOLOv8 model)
- A webcam is required for real-time testing.
- Flask app runs locally at `http://127.0.0.1:5000/`

---

## Future Enhancements
- Add more gestures (e.g., head tilt, eye roll)
- Support multi-modal biometrics (voice, fingerprint)
- Host the system on a cloud platform

---

Developed by: Krupali Shinde  
CS 668 - Analytics Capstone Project
