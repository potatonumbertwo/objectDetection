from ultralytics import YOLO
import cv2

model = YOLO('../Yolo_weight/yolov8l.pt')
# results = model('images/wriothesley.png', show=True)
# results = model('images/neuvillette.png', show=True)
results = model('images/2.jpeg', show=True)
cv2.waitKey(0)
