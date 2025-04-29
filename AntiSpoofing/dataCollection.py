from cvzone.FaceDetectionModule import FaceDetector
import cv2
import cvzone
from time import time
import os

# Configuration
classID = 0  # 0 = fake, 1 = real
outputFolderPath = 'Anti-Spoofing/Dataset/DataCollect'
confidence = 0.8
save = True
blurThreshold = 35
offsetPercentageW = 10
offsetPercentageH = 20
camWidth, camHeight = 640, 480
floatingPoint = 6
debug = True  # Set to True to print blur/brightness info

# Ensure output folder exists
os.makedirs(outputFolderPath, exist_ok=True)

# Initialize webcam and detector
cap = cv2.VideoCapture(1)
cap.set(3, camWidth)
cap.set(4, camHeight)

# Optional: Try to adjust exposure (depends on camera support)
# cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
# cap.set(cv2.CAP_PROP_EXPOSURE, -5)

detector = FaceDetector(minDetectionCon=0.5, modelSelection=0)

while True:
    success, img = cap.read()
    if not success:
        print("Camera read failed.")
        break

    imgOut = img.copy()
    img, bboxs = detector.findFaces(img, draw=False)

    listBlur = []
    listInfo = []

    if bboxs:
        for bbox in bboxs:
            x, y, w, h = bbox["bbox"]
            score = bbox["score"][0]

            if score > confidence:
                # Expand bounding box with offsets
                offsetW = (offsetPercentageW / 100) * w
                offsetH = (offsetPercentageH / 100) * h
                x = max(int(x - offsetW), 0)
                y = max(int(y - offsetH * 3), 0)
                w = int(w + offsetW * 2)
                h = int(h + offsetH * 3.5)

                imgFace = img[y: y + h, x: x + w]
                if imgFace.size == 0:
                    print("Empty face image – skipping")
                    continue

                cv2.imshow("Face", imgFace)

                # Analyze blur and brightness
                blurValue = int(cv2.Laplacian(imgFace, cv2.CV_64F).var())
                grayFace = cv2.cvtColor(imgFace, cv2.COLOR_BGR2GRAY)
                brightness = cv2.mean(grayFace)[0]

                dynamicThreshold = 8 if brightness < 50 else blurThreshold
                isFocused = blurValue > dynamicThreshold
                listBlur.append(isFocused)

                if debug:
                    print(f"Blur: {blurValue}, Brightness: {int(brightness)}, Focused: {isFocused}")
                if not isFocused:
                    print("Low light or blurry image – consider improving lighting.")

                # Normalize coordinates for YOLO label
                ih, iw, _ = img.shape
                xc, yc = x + w / 2, y + h / 2
                xcn, ycn = round(xc / iw, floatingPoint), round(yc / ih, floatingPoint)
                wn, hn = round(w / iw, floatingPoint), round(h / ih, floatingPoint)
                listInfo.append(f"{classID} {min(xcn,1)} {min(ycn,1)} {min(wn,1)} {min(hn,1)}\n")

                # Draw rectangle + score info
                cv2.rectangle(imgOut, (x, y, w, h), (255, 0, 0), 3)
                cvzone.putTextRect(imgOut, f'Score: {int(score*100)}% Blur: {blurValue}', (x, y - 10),
                                   scale=2, thickness=3)

                # Class info
                cvzone.putTextRect(imgOut, f'Class: {"Real" if classID==1 else "Fake"}', (x, y + h + 30),
                                   scale=1, thickness=1)

    # Save image and label if at least one face detected
    if save and listBlur != []:
        timeNow = str(time()).replace('.', '')
        imagePath = f"{outputFolderPath}/{timeNow}.jpg"
        textPath = f"{outputFolderPath}/{timeNow}.txt"

        # Save even if blurry (useful for training on bad data too)
        cv2.imwrite(imagePath, img)
        print(f" Saved: {imagePath}")

        with open(textPath, 'a') as f:
            f.writelines(listInfo)

    # Show output
    cv2.imshow("Image", imgOut)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
