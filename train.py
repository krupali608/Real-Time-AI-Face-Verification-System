from ultralytics import YOLO

model = YOLO('yolov8n.pt')

def main():
    model.train(data='Anti-Spoofing/Dataset/splitData/dataOffline.yaml', epochs=50)

if __name__ == '__main__':
    main()
