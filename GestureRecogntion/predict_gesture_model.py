import cv2
import dlib
import numpy as np
import time
import random
from collections import Counter
from keras.models import load_model
from imutils import face_utils
import threading

# Load model and labels
model = load_model("gesture_classifier_model.h5")
class_labels = ['blink', 'smile']
CONFIDENCE_THRESHOLD = 0.90

# Load dlib predictor
predictor_path = "Anti-Spoofing/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# Set up camera with smaller resolution for better speed
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
if not cap.isOpened():
    print("Unable to access webcam.")
    exit()
print("Camera started.")
time.sleep(1)

# Thread-safe frame grabbing
def grab_frames():
    global current_frame, ret
    while cap.isOpened():
        ret, current_frame = cap.read()
        if not ret:
            break

current_frame = None
ret = False
thread = threading.Thread(target=grab_frames)
thread.daemon = True
thread.start()

# Utility functions (modified to return bounding box too)
def extract_eye_region(frame, shape_np):
    left_eye = shape_np[36:42]
    right_eye = shape_np[42:48]
    (lx, ly, lw, lh) = cv2.boundingRect(left_eye)
    (rx, ry, rw, rh) = cv2.boundingRect(right_eye)
    x1 = max(0, min(lx, rx) - 10)
    y1 = max(0, min(ly, ry) - 10)
    x2 = min(frame.shape[1], max(lx + lw, rx + rw) + 10)
    y2 = min(frame.shape[0], max(ly + lh, ry + rh) + 10)
    eye_region = frame[y1:y2, x1:x2]
    return eye_region, (x1, y1, x2, y2)

def extract_mouth_region(frame, shape_np):
    mouth = shape_np[48:68]
    (mx, my, mw, mh) = cv2.boundingRect(mouth)
    x1 = max(0, mx - 15)
    y1 = max(0, my - 15)
    x2 = min(frame.shape[1], mx + mw + 15)
    y2 = min(frame.shape[0], my + mh + 15)
    mouth_region = frame[y1:y2, x1:x2]
    return mouth_region, (x1, y1, x2, y2)

# Get initial frame and command
time.sleep(1)
while current_frame is None:
    time.sleep(0.1)

command = random.choice(class_labels)
print(f"Perform this gesture: {command}")
cv2.putText(current_frame, f"Command: {command}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
cv2.imshow("Gesture Verification", current_frame)
cv2.waitKey(1500)

# Prediction loop
predictions = []
frame_count = 10
print("Capturing... Hold your gesture")
start_time = time.time()
frame_index = 0

while len(predictions) < frame_count and time.time() - start_time < 5:
    if current_frame is None:
        continue
    frame_index += 1
    if frame_index % 2 != 0:
        continue  # Skip every alternate frame

    frame = current_frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    if len(rects) == 0:
        continue

    shape = predictor(gray, rects[0])
    shape_np = face_utils.shape_to_np(shape)

    try:
        if command == 'blink':
            crop, (x1, y1, x2, y2) = extract_eye_region(frame, shape_np)
        else:
            crop, (x1, y1, x2, y2) = extract_mouth_region(frame, shape_np)
    except:
        continue

    if crop.size == 0 or crop.shape[0] < 30 or crop.shape[1] < 30:
        continue

    img_resized = cv2.resize(crop, (224, 224))
    img_normalized = img_resized.astype("float32") / 255.0
    img_input = np.expand_dims(img_normalized, axis=0)

    preds = model.predict(img_input, verbose=0)
    pred_index = np.argmax(preds)
    predictions.append((class_labels[pred_index], preds[0][pred_index]))

    # Draw command, prediction, and bounding box
    cv2.putText(frame, f"Command: {command}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.putText(frame, f"Prediction: {class_labels[pred_index]} ({preds[0][pred_index]*100:.1f}%)", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (100, 255, 100), 2)  #  draw bounding box here
    cv2.imshow("Gesture Verification", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Final decision
labels_only = [label for label, _ in predictions]
confidences = [conf for _, conf in predictions]
most_common_label, count = Counter(labels_only).most_common(1)[0]
best_confidence = max([conf for label, conf in predictions if label == most_common_label])

final_frame = current_frame.copy() if current_frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)
cv2.putText(final_frame, f"Command: {command}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

if most_common_label == command and best_confidence >= CONFIDENCE_THRESHOLD:
    result_text = f"Verified: {most_common_label} ({best_confidence*100:.1f}%)"
    color = (0, 255, 0)
elif most_common_label == command:
    result_text = f"Retry: Low confidence ({best_confidence*100:.1f}%)"
    color = (0, 255, 255)
else:
    result_text = f"Spoof Detected: {most_common_label} ({best_confidence*100:.1f}%)"
    color = (0, 0, 255)

cv2.putText(final_frame, result_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
cv2.imshow("Gesture Verification", final_frame)
cv2.waitKey(4000)

cap.release()
cv2.destroyAllWindows()
