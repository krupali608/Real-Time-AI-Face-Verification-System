from pathlib import Path
from flask import Flask, render_template, Response, request, jsonify
import cv2, dlib, numpy as np, time, os
from collections import Counter
from tensorflow.keras.models import load_model
from imutils import face_utils
from ultralytics import YOLO

base_dir = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder=os.path.join(base_dir, 'templates'), static_folder=os.path.join(base_dir, 'static'))

# Load models
spoof_model = YOLO(os.path.join(base_dir, 'models/n_version_1.pt'))
gesture_model = load_model(os.path.join(base_dir, 'gesture_classifier_model.h5'))
predictor_path = os.path.join(base_dir, 'shape_predictor_68_face_landmarks.dat')

# Configs
CONFIDENCE_THRESHOLD = 0.90
classNames = ["fake", "real"]
gesture_command = 'blink'

# Dlib and camera
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Globals
real_face_detected = False
fail_count = 0

def extract_eye_region(frame, shape_np):
    left_eye = shape_np[36:42]
    right_eye = shape_np[42:48]
    lx, ly, lw, lh = cv2.boundingRect(left_eye)
    rx, ry, rw, rh = cv2.boundingRect(right_eye)
    x1, y1 = min(lx, rx), min(ly, ry)
    x2, y2 = max(lx + lw, rx + rw), max(ly + lh, ry + rh)
    return frame[y1-10:y2+10, x1-10:x2+10]

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
def gen_frames():
    global real_face_detected
    while True:
        success, frame = camera.read()
        if not success:
            break

        results = spoof_model(frame, stream=True, verbose=False)
        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = f"{classNames[cls]} {conf*100:.1f}%"
                color = (0, 255, 0) if classNames[cls] == 'real' else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                if conf > 0.8 and classNames[cls] == 'real':
                    real_face_detected = True

            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/verify', methods=['POST'])
def verify():
    global real_face_detected, fail_count
    if fail_count >= 3:
        real_face_detected = False  # Reset on block
        return jsonify({'status': 'blocked', 'message': 'Too many failed attempts. Access denied.'})
    
    if not real_face_detected:
        fail_count += 1
        real_face_detected = False  # Reset after spoof detection
        return jsonify({'status': 'spoof', 'message': 'Spoof Detected. Retry again.'})

    response = jsonify({'status': 'command', 'command': gesture_command, 'message': 'Perform this gesture: blink'})
    real_face_detected = False  # Reset after command is issued (force new check later)
    return response

@app.route('/gesture_verify', methods=['POST'])
def gesture_verify():
    global fail_count
    predictions = []
    frame_count = 20
    start_time = time.time()

    while len(predictions) < frame_count and time.time() - start_time < 5:
        ret, frame = camera.read()
        if not ret:
            continue

        results = spoof_model(frame, stream=True, verbose=False)
        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                if conf > 0.8 and classNames[cls] == 'fake':
                    fail_count += 1
                    return jsonify({'status': 'spoof', 'message': 'Spoof Detected during gesture. Retry again.'})

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)
        if not rects:
            continue

        shape = predictor(gray, rects[0])
        shape_np = face_utils.shape_to_np(shape)
        try:
            crop = extract_eye_region(frame, shape_np)
        except:
            continue

        if crop.size == 0:
            continue

        img_input = np.expand_dims(cv2.resize(crop, (224, 224)).astype("float32") / 255.0, axis=0)
        preds = gesture_model.predict(img_input, verbose=0)
        pred_index = np.argmax(preds)
        predictions.append(('blink', preds[0][pred_index]))

    if not predictions:
        fail_count += 1
        return jsonify({'status': 'retry', 'message': 'No valid gesture detected.'})

    label_counts = Counter([label for label, _ in predictions])
    most_common_label, count = label_counts.most_common(1)[0]
    blink_ratio = count / frame_count
    best_conf = max([conf for label, conf in predictions if label == most_common_label])

    if most_common_label == 'blink' and blink_ratio >= 0.7 and best_conf >= CONFIDENCE_THRESHOLD:
        fail_count = 0
        return jsonify({'status': 'verified', 'message': 'Verified successfully!'})
    else:
        fail_count += 1
        real_face_detected = False
        return jsonify({'status': 'retry', 'message': 'Gesture not confident. Please retry.'})

if __name__ == '__main__':
    app.run(debug=True)
