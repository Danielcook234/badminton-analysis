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
                 court_w_m=6.1, court_h_m=13.4,):

        # Parameters
        self.video_path = video_path
        self.court_w_m = court_w_m
        self.court_h_m = court_h_m
        self.model_path = model_path
        self.subtractor = cv2.createBackgroundSubtractorKNN(50,50,0)

        self.model = YOLO(self.model_path)

        # Load video
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open video.")

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            results = self.model.predict(source=frame, verbose = False)

            for r in results:
                for box in r.boxes:
                    
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls_ids = int(box.cls[0])

                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                    colour_box = (0,255,0)
                    cv2.rectangle(frame, (x1,y1), (x2,y2), colour_box, 2)


            frame = cv2.resize(frame, (1400, 750))

            #frame in hls colour format
            hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)

            #grayscale frame
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

            roi = cv2.bitwise_or(mask, bg_remove)

            cv2.imshow('ROI', roi)

            lines = self.find_court_lines(roi)

            for pt1,pt2 in lines.values():
                cv2.line(frame, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)

            cv2.imshow("badminton", frame)


            if cv2.waitKey(25) == 27:
                break

        self.cleanup()

    def find_court_lines(self, bg_frame):
        dst = cv2.Canny(bg_frame, 50, 200, None, 3)

        lines = cv2.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 10)

        line_dict = {}

        if lines is not None:
            for i in range(0, len(lines)):
                l = lines[i][0]
                line_dict[i] = ((l[0],l[1]), (l[2],l[3]))
            
        return line_dict


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