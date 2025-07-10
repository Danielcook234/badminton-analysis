import cv2
from ultralytics import YOLO
from collections import deque

model = YOLO('runs/detect/shuttlecock_yolov8n2/weights/best.pt')

cap = cv2.VideoCapture("rally1.mp4")

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

cv2.namedWindow("Shuttlecock Detection", cv2.WINDOW_NORMAL)

trajectory = deque(maxlen=30)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(source=frame, verbose=False)
    
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = int((x1+x2) / 2)
            cy = int ((y1 + y2) / 2)
            if float(box.conf[0]) > 0.5:
                trajectory.append((cx,cy))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, 'shuttlecock', (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            
    for i in range(1,len(trajectory)):
        cv2.line(frame,trajectory[i-1],trajectory[i], (255,0,0),2)

    cv2.imshow("Shuttlecock Detection", frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
