import cv2
from ultralytics import YOLO
from collections import deque
import numpy as np

def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked at: {x}, {y}")

def project_to_court(u, v, camera_matrix, rvec, tvec):
    """Back-project image coords (u,v) onto the floor Z=0 using camera pose."""
    R, _ = cv2.Rodrigues(rvec)
    K_inv = np.linalg.inv(camera_matrix)

    uv_hom = np.array([u, v, 1.0], dtype=np.float32)
    ray_dir = K_inv @ uv_hom
    ray_dir = R.T @ ray_dir
    cam_centre = -R.T @ tvec

    t = -cam_centre[2] / ray_dir[2]  # intersection with Z=0
    world_point = cam_centre + t * ray_dir
    return world_point.flatten()[:3]

# ---------- Main ----------
if __name__ == "__main__":

    class_names = ['Player 1', 'Player 2', 'shuttle']

    # Court corners in video frame
    image_points = np.array([
        [416, 1012],  # bottom-left
        [1506, 1012], # bottom-right
        [1311, 582],  # top-right
        [606, 582]    # top-left
    ], dtype=np.float32)

    # Court corners in top-down view (warped)
    c_width, c_height = 1340, 900
    pts_dst = np.array([[0, c_height], [c_width, c_height], [c_width, 0], [0, 0]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(image_points, pts_dst)

    # Camera intrinsics (approximate)
    focal_length = 1500
    centre = (c_width/2, c_height/2)
    camera_matrix = np.array([
        [focal_length, 0, centre[0]],
        [0, focal_length, centre[1]],
        [0, 0, 1]
    ], dtype=np.float32)
    dist_coeffs = np.zeros((4,1))

    # SolvePnP once to get camera pose
    object_points = np.array([
        [0.0, 0.0, 0.0],
        [6.1, 0.0, 0.0],
        [6.1, 13.4, 0.0],
        [0.0, 13.4, 0.0]
    ], dtype=np.float32)
    success, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)

    # Load YOLO and video
    model = YOLO('runs/detect/shuttlecock_yolov8n4/weights/best.pt')
    cap = cv2.VideoCapture("rally1.mp4")
    if not cap.isOpened():
        raise RuntimeError("Could not open video.")

    cv2.namedWindow("Shuttlecock Detection", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('Shuttlecock Detection', click_event)

    trajectory = deque(maxlen=15)
    shown_trajectory = deque(maxlen=30)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        warped_frame = cv2.warpPerspective(frame, H, (c_width, c_height))
        results = model.predict(source=frame, verbose=False)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                label = class_names[cls_id]

                # Bottom-center of bbox
                cx = int((x1 + x2)/2)
                cy = int(y2)

                if label == 'shuttle' and float(box.conf[0]) > 0.5:
                    # 3D back-projection to court
                    wx_world, wy_world, wz_world = project_to_court(cx, cy, camera_matrix, rvec, tvec)

                    # Clamp to court size
                    wx_world = np.clip(wx_world, 0, 6.1)
                    wy_world = np.clip(wy_world, 0, 13.4)

                    # Map to top-down display
                    disp_x = int(wx_world / 6.1 * c_width)
                    disp_y = int(c_height - (wy_world / 13.4 * c_height))

                    cv2.circle(warped_frame, (disp_x, disp_y), 5, (0,255,255), -1)

                    trajectory.append((cx, cy))
                    shown_trajectory.append((cx, cy))

                else:
                    # Players: bottom of bbox → warp using homography
                    point = np.array([[[cx, cy]]], dtype=np.float32)
                    warped_c = cv2.perspectiveTransform(point, H)
                    wx, wy = warped_c[0][0]
                    colour = (0,0,255)  # player red
                    cv2.circle(warped_frame, (int(wx), int(wy)), 5, colour, -1)

                # Draw bbox on original video
                colour_box = (0, 255, 0) if label=='shuttle' else (255,0,0)
                cv2.rectangle(frame, (x1,y1), (x2,y2), colour_box, 2)
                cv2.putText(frame, label, (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour_box, 2)

        # Draw shuttle trajectory on original video
        for i in range(1, len(shown_trajectory)):
            cv2.line(frame, shown_trajectory[i-1], shown_trajectory[i], (255,0,0), 2)

        # Display frames
        cv2.imshow("Shuttlecock Detection", frame)
        cv2.imshow("Warped perspective", warped_frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()