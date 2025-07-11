import cv2
from ultralytics import YOLO
from collections import deque
import numpy as np

last_curve_sign = [None]

def detect_player_hit(trajectory, tolerance = 25, fit_points = 6):
    if len(trajectory) < fit_points + 1:
        return False, None
    
    recent = list(trajectory)[-fit_points-1:-1]
    new_point = trajectory[-1]

    xs = [p[0] for p in recent]
    ys = [p[1] for p in recent]

    try:
        coeffs = np.polyfit(xs,ys,2)
        a,b,c = coeffs
        predicted_y = a * new_point[0]**2 + b * new_point[0] + c
        actual_y = new_point[1]
        error = abs(predicted_y - actual_y)

        if error > tolerance:
            return True, coeffs
        else:
            return False, coeffs

    except np.RankWarning:
        return False, None

if __name__ == "__main__":

    model = YOLO('runs/detect/shuttlecock_yolov8n2/weights/best.pt')

    cap = cv2.VideoCapture("rally1.mp4")

    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    cv2.namedWindow("Shuttlecock Detection", cv2.WINDOW_NORMAL)

    trajectory = deque(maxlen=15)

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

        player_hit, curve = detect_player_hit(trajectory)

        if player_hit:
            cv2.putText(frame, "Shuttle Hit!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX,
                1.2, (0, 0, 255), 3)

            # Optionally log or visualize
            with open("detection.txt", "a") as f:
                f.write(f"HIT DETECTED due to trend break. Coeffs: {curve}\n")

            # Reset trajectory to start new trend
            trajectory = deque([trajectory[-1]], maxlen=15)

        cv2.imshow("Shuttlecock Detection", frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
