from ultralytics import YOLO
import cv2
import cvzone
import math

# number of camera
# cap = cv2.VideoCapture(0)
# setting the capture size
# cap.set(3, 640)

# video
# cap = cv2.VideoCapture('avsk.mp4')
# cap = cv2.VideoCapture('mona.mp4')
# cap = cv2.VideoCapture('lingtuo.mp4')

# cap = cv2.VideoCapture('haha.mp4')
# cap = cv2.VideoCapture('alhaitham.mp4')
# cap = cv2.VideoCapture('cyno.mp4')
# cap = cv2.VideoCapture('erchuang.mp4')
cap = cv2.VideoCapture('nave.mp4')

# build model
model = YOLO('../Yolo_weight/genshin.pt')
classnames = ['Albedo', 'Alhaitham', 'Aloy', 'Amber', 'Itto', 'Baizhu', 'Barbara', 'Beidou', 'Bennett', 'Candace',
              'Charlotte', 'Chongyun', 'Collei', 'Cyno', 'Dainsleif', 'Dehya', 'Diluc', 'Diona', 'Dori', 'Eula',
              'Faruzan',
              'Fischl', 'Freminet', 'Furina', 'Ganyu', 'Gorou', 'HuTao', 'Jean', 'Kazuha', 'Kaeya', 'Ayaka', 'Ayato',
              'Kaveh',
              'Keqing', 'Kirara', 'Klee', 'KujoSara', 'KukiShinobu', 'Layla', 'Lisa', 'Lynette', 'Lyney', 'Mika',
              'Mona',
              'Nahida', 'Neuvillette', 'Nilou', 'Ningguang', 'Noelle', 'Paimon', 'Qiqi', 'RaidenShogun', 'Razor',
              'Rosaria',
              'Kokomi', 'Sayu', 'Shenhe', 'Heizou', 'Sucrose', 'Tartaglia', 'Thoma', 'Tighnari', 'Aether', 'Lumine',
              'Venti',
              'Wanderer', 'Wriothesley', 'Xiangling', 'Xiao', 'Xingqiu', 'Xinyan', 'Miko', 'Yaoyao', 'Yanfei', 'Yelan',
              'Yoimiya', 'YunJin', 'Zhongli']
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

            # conf
            conf = math.ceil((box.conf[0] * 100)) / 100

            # class name
            cls = int(box.cls[0])

            # display the class name and its conf
            cvzone.putTextRect(img, f'{classnames[cls]} {conf}',
                               (max(0, x1), max(30, y1)), scale=1, thickness=2)

    cv2.imshow('image', img)
    cv2.waitKey(1)
