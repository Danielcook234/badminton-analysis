import cv2
import numpy as np
import csv
from ultralytics import YOLO

# ---------------- Helper ----------------
def click_event(event, x, y, flags, param):
    """Mouse click event for court labeling."""
    if event == cv2.EVENT_LBUTTONDOWN:
        param['click'] = (x, y)


# ---------------- Main Class ----------------
class ShuttleCourtMapper:
    def __init__(self, video_path, model_path, court_img_path,
                 court_w_m=6.1, court_h_m=13.4,
                 csv_output="shuttle_ground_truth.csv"):

        # Parameters
        self.video_path = video_path
        self.model_path = model_path
        self.court_img_path = court_img_path
        self.court_w_m = court_w_m
        self.court_h_m = court_h_m
        self.csv_output = csv_output

        # Load model and video
        self.model = YOLO(self.model_path)
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open video.")

        # Load court diagram
        self.court_img = cv2.imread(self.court_img_path)
        if self.court_img is None:
            raise RuntimeError("Could not load court image.")
        self.court_h, self.court_w = self.court_img.shape[:2]

        # Open CSV
        self.csv_file = open(self.csv_output, 'w', newline='')
        self.writer = csv.writer(self.csv_file)
        self.writer.writerow(['frame', 'cx', 'cy', 'court_X', 'court_Y'])

    def run(self):
        """Main loop for collecting shuttle-to-court labels."""
        frame_idx = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            results = self.model.predict(source=frame, verbose=False)

            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    if cls_id != 2:  # only shuttle
                        continue

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                    # Show detection
                    display_frame = frame.copy()
                    cv2.circle(display_frame, (cx, cy), 6, (0, 255, 255), -1)
                    cv2.putText(display_frame, f"Frame {frame_idx}", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                    cv2.imshow("Shuttle Detection", display_frame)
                    cv2.waitKey(1)

                    # Court click loop
                    click_state = {'click': None}
                    cv2.namedWindow("Court", cv2.WINDOW_NORMAL)
                    cv2.setMouseCallback("Court", click_event, click_state)

                    while click_state['click'] is None:
                        court_copy = self.court_img.copy()
                        cv2.putText(court_copy, "Click shuttle landing", (50, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                        cv2.imshow("Court", court_copy)
                        cv2.waitKey(1)

                    click_x, click_y = click_state['click']

                    # Convert to meters
                    court_X = (click_x / self.court_w) * self.court_w_m
                    court_Y = ((self.court_h - click_y) / self.court_h) * self.court_h_m

                    # Save row
                    self.writer.writerow([frame_idx, cx, cy, court_X, court_Y])
                    print(f"Frame {frame_idx}: shuttle ({cx},{cy}) -> court ({court_X:.2f},{court_Y:.2f})")

            frame_idx += 1

        self.cleanup()

    def cleanup(self):
        """Release resources cleanly."""
        self.cap.release()
        self.csv_file.close()
        cv2.destroyAllWindows()


# ---------------- Entrypoint ----------------
if __name__ == "__main__":
    mapper = ShuttleCourtMapper(
        video_path="rally1.mp4",
        model_path="runs/detect/shuttlecock_yolov8n4/weights/best.pt",
        court_img_path="court_topdown.png",
        court_w_m=6.1,
        court_h_m=13.4,
        csv_output="shuttle_ground_truth.csv"
    )
    mapper.run()