import cv2
import numpy as np
import math
from ultralytics import YOLO

def click_event(event, x, y, flags, param):
    """Mouse click event for court labeling."""
    if event == cv2.EVENT_LBUTTONDOWN:
        param['click'] = (x, y)

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

        # Load video
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open video.")
        
        self.last_results = []
        self.last_lines = {}

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
                l = lines[i][0]
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

            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(proc_frame, (x1,y1), (x2,y2), (0,255,0), 2)


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