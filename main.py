import cv2
from ultralytics import YOLO
from collections import deque
import numpy as np

last_curve_sign = [None]

def detect_player_hit(trajectory, speed_threshold = 20, angle_change_threshold = 120):
    #not enough points
    if len(trajectory) < 3:
        return False, None
    
    #get last 3 points
    (x1,y1),(x2,y2),(x3,y3) = trajectory[-3], trajectory[-2],trajectory[-1]

    #compute velocity vectors
    v1 = np.array([x2-x1,y2-y1])
    v2 = np.array([x3-x2,y3-y2])

    #speeds
    speed1 = np.linalg.norm(v1)
    speed2 = np.linalg.norm(v2)

    with open("detection.txt", "a") as f:
        f.write(f"last 3 points: {(x1,y1)}, {(x2,y2)}, {(x3,y3)}\n")
        f.write(f"v1: {v1}, v2: {v2}\n")
        f.write(f"speed1: {speed1}, speed2: {speed2}\n")

    #angle between movement vectors
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return False, None
    
    cosine_angle = np.dot(v1,v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.degrees(np.arccos(np.clip(cosine_angle,-1.0,1.0)))

    #detect a hit
    hit_detected = abs(speed2-speed1) > speed_threshold or angle > angle_change_threshold
    if not hit_detected:
        return False, None
    
    hitter = "bottom player" if v1[1] < 0 else "top player"

    return True, hitter

if __name__ == "__main__":

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

        player_hit, hitter = detect_player_hit(trajectory)

        with open("detection.txt", "a") as f:
            f.write(f"hit_detected: {player_hit}\n")
            f.write(f"hitter: {hitter}\n\n")

        if player_hit:
            cv2.putText(frame, f"{hitter} Hit!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX,
                1.2, (0, 0, 255), 3)

        cv2.imshow("Shuttlecock Detection", frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
