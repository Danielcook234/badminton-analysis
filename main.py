import cv2
import numpy as np
import csv
from ultralytics import YOLO

def click_event(event, x, y, flags, param):
    """Mouse click event for court labeling."""
    if event == cv2.EVENT_LBUTTONDOWN:
        param['click'] = (x, y)

class ShuttleCourtMapper:
    def __init__(self, video_path,
                 court_w_m=6.1, court_h_m=13.4,):

        # Parameters
        self.video_path = video_path
        self.court_w_m = court_w_m
        self.court_h_m = court_h_m
        self.subtractor = cv2.createBackgroundSubtractorKNN(50,50,0)

        # Load video
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open video.")

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

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

            cv2.imshow("badminton", frame)

            if cv2.waitKey(25) == 27:
                break

        self.cleanup()

    def cleanup(self):
        """Release resources cleanly."""
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    mapper = ShuttleCourtMapper(
        video_path="rally1.mp4",
        court_w_m=6.1,
        court_h_m=13.4,
    )
    mapper.run()