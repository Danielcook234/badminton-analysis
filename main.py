import cv2
import numpy as np
import math
from collections import deque
from ultralytics import YOLO

def click_event(event, x, y, flags, param):
    """Mouse click event for court labeling."""
    if event == cv2.EVENT_LBUTTONDOWN:
        param['click'] = (x, y)

def detect_player_hit(trajectory, tolerance=25, fit_points=6):
    """Fit a quadratic to recent shuttle points and flag a hit when the newest
    point breaks that trend (i.e. the shuttle changed direction)."""
    if len(trajectory) < fit_points + 1:
        return False, None

    recent = list(trajectory)[-fit_points-1:-1]
    new_point = trajectory[-1]

    xs = [p[0] for p in recent]
    ys = [p[1] for p in recent]

    try:
        coeffs = np.polyfit(xs, ys, 2)
        a, b, c = coeffs
        predicted_y = a * new_point[0]**2 + b * new_point[0] + c
        error = abs(predicted_y - new_point[1])
        return error > tolerance, coeffs
    except np.RankWarning:
        return False, None

class ShuttleCourtMapper:
    def __init__(self, video_path, model_path,
                 court_w_m=6.1, court_h_m=13.4,
                 yolo_interval=3, line_interval=5,
                 proc_size=(700,375)):

        # Parameters
        self.video_path = video_path
        self.court_w_m = court_w_m
        self.court_h_m = court_h_m
        self.model_path = model_path

        self.yolo_interval = yolo_interval
        self.line_interval = line_interval
        self.proc_size = proc_size

        self.subtractor = cv2.createBackgroundSubtractorKNN(50,50,0)
        self.model = YOLO(self.model_path)
        self.class_names = ['Player 1', 'Player 2', 'shuttle']

        # Load video
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open video.")

        self.last_results = []
        self.last_lines = {}

        # Shuttle trajectory, smoothed/interpolated every frame via Kalman filter
        # so gaps between YOLO frames (yolo_interval) don't show up as jerky jumps.
        self.trajectory = deque(maxlen=15)
        self.shown_trajectory = deque(maxlen=30)
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 1.0
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1
        self.kalman_initialized = False

    def preprocess_for_lines(self, frame):
        hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #white mask to detect white lines
        lower_white = np.uint8([0, 200, 0])
        upper_white = np.uint8([255, 255, 255])
        white_mask = cv2.inRange(hls, lower_white, upper_white)

        #yellow mask
        lower_yellow = np.uint8([10, 0, 100])
        upper_yellow = np.uint8([40, 255, 255])
        yellow_mask = cv2.inRange(hls, lower_yellow, upper_yellow)

        #combine mask
        mask = cv2.bitwise_or(white_mask, yellow_mask)

        #remove background from gray frame
        bg_remove = self.subtractor.apply(gray)

        return cv2.bitwise_or(mask, bg_remove)
    
    def find_court_lines(self, bg_frame):

        dst = cv2.Canny(bg_frame, 50, 200, None, 3)
        lines = cv2.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 10)

        line_dict = {}
        if lines is not None:
            for i in range(0, len(lines)):
                l = lines[i].reshape(4)
                line_dict[i] = ((l[0],l[1]), (l[2],l[3]))

        return line_dict

    def run(self):
        frame_id = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            #downscale for processing
            proc_frame = cv2.resize(frame,self.proc_size)

            #yolo every N frames
            if frame_id % self.yolo_interval == 0:
                results = self.model.predict(source=proc_frame, verbose = False)
                self.last_results = results
            else:
                results = self.last_results

            shuttle_measurement = None
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(proc_frame, (x1,y1), (x2,y2), (0,255,0), 2)

                    label = self.class_names[int(box.cls[0])]
                    if label == 'shuttle' and float(box.conf[0]) > 0.3:
                        shuttle_measurement = ((x1 + x2) / 2, (y1 + y2) / 2)

            # Predict every frame regardless of yolo_interval, so the trajectory
            # stays smooth even on frames with no fresh YOLO detection; correct
            # against the real detection only on frames that have one.
            if shuttle_measurement is not None and not self.kalman_initialized:
                cx, cy = shuttle_measurement
                self.kalman.statePre = np.array([[cx], [cy], [0], [0]], np.float32)
                self.kalman.statePost = np.array([[cx], [cy], [0], [0]], np.float32)
                self.kalman_initialized = True

            if self.kalman_initialized:
                predicted = self.kalman.predict()
                if shuttle_measurement is not None:
                    cx, cy = shuttle_measurement
                    measurement = np.array([[np.float32(cx)], [np.float32(cy)]])
                    predicted = self.kalman.correct(measurement)

                point = (int(predicted[0][0]), int(predicted[1][0]))
                self.trajectory.append(point)
                self.shown_trajectory.append(point)

                for i in range(1, len(self.shown_trajectory)):
                    cv2.line(proc_frame, self.shown_trajectory[i-1], self.shown_trajectory[i], (255,0,0), 2)

                player_hit, curve = detect_player_hit(self.trajectory)
                if player_hit:
                    cv2.putText(proc_frame, "Shuttle Hit!", (30, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                    with open("detection.txt", "a") as f:
                        f.write(f"HIT DETECTED due to trend break. Coeffs: {curve}\n")
                    self.trajectory = deque([self.trajectory[-1]], maxlen=15)

            if frame_id % self.line_interval == 0:
                roi = self.preprocess_for_lines(proc_frame)
                self.last_lines = self.find_court_lines(roi)
            lines = self.find_court_lines(roi)

            for pt1,pt2 in lines.values():
                cv2.line(proc_frame, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)

            display = cv2.resize(proc_frame, (1400,750))

            cv2.imshow("badminton", display)


            if cv2.waitKey(25) == 27:
                break

            frame_id += 1

        self.cleanup()

    def cleanup(self):
        """Release resources cleanly."""
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    mapper = ShuttleCourtMapper(
        video_path="rally1.mp4",
        model_path="runs/detect/shuttlecock_yolov8n8/weights/best.pt",
        court_w_m=6.1,
        court_h_m=13.4,
    )
    mapper.run()