from ultralytics import YOLO
import cv2
import cvzone
import math

# number of camera
# cap = cv2.VideoCapture(0)
# setting the capture size
# cap.set(3, 640)

# video
cap = cv2.VideoCapture('../videos/genshin.mp4')

# build model
model = YOLO('../Yolo_weight/yolov8n.pt')
classnames = []
while True:
    success, img = cap.read()
    results = model(img, stream=True)

    # getting bounding box for each image
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)

            #confidence
            confidence = math.ceil((box.conf[0]*100))/100

            # class name
            class_name = int(box.cls[0])

            # display the class name and its confidence
            cvzone.putTextRect(img, f'{classnames}{confidence}',
                               (max(0, x1), max(30, y1)), scale=0.5, thickness=1)



    cv2.imshow('image', img)
    cv2.waitKey(1)


