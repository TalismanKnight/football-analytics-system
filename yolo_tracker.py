from ultralytics import YOLO
import cv2
import numpy as np
import os

# ----------------------------
# Paths
# ----------------------------
video_path = os.path.join("videos", "match.mp4")
output_path = os.path.join("output", "output.mp4")

# ----------------------------
# Load YOLO model
# ----------------------------
model = YOLO("yolov8n.pt")

# ----------------------------
# Open video
# ----------------------------
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 30

resize_width = 640
resize_height = 360

# ----------------------------
# Video Writer
# ----------------------------
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (resize_width, resize_height))

# ----------------------------
# Tracking Variables
# ----------------------------
trail_points = []
max_trail_length = 25

prev_center = None
speed_kmh = 0
pixel_to_meter = 0.004  # calibrated value

heatmap = None

# ----------------------------
# Main Loop
# ----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (resize_width, resize_height))

    # Initialize heatmap once
    if heatmap is None:
        heatmap = np.zeros((resize_height, resize_width), dtype=np.float32)

    results = model(frame)

    ball_found = False

    for r in results:
        boxes = r.boxes

        for box in boxes:
            cls = int(box.cls[0])

            # Class 32 = sports ball
            if cls == 32:
                ball_found = True

                x1, y1, x2, y2 = box.xyxy[0]
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                center = (cx, cy)

                # ---- Update Heatmap ----
                if 0 <= cy < resize_height and 0 <= cx < resize_width:
                    heatmap[cy, cx] += 1

                # ---- Speed Calculation ----
                if prev_center is not None:
                    dx = center[0] - prev_center[0]
                    dy = center[1] - prev_center[1]
                    pixel_dist = np.sqrt(dx**2 + dy**2)

                    meter_dist = pixel_dist * pixel_to_meter
                    speed_mps = meter_dist * fps
                    new_speed = speed_mps * 3.6

                    # Smooth speed
                    speed_kmh = 0.8 * speed_kmh + 0.2 * new_speed

                prev_center = center

                # ---- Store Trail ----
                trail_points.append(center)
                if len(trail_points) > max_trail_length:
                    trail_points.pop(0)

                # Draw bounding box
                cv2.rectangle(frame,
                              (int(x1), int(y1)),
                              (int(x2), int(y2)),
                              (0, 255, 0),
                              2)

                cv2.putText(frame,
                            "BALL",
                            (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 0),
                            2)

    if not ball_found:
        trail_points.clear()
        prev_center = None

    # ---- Draw Smooth Trail ----
    for i in range(1, len(trail_points)):
        thickness = max(1, int(5 * (i / len(trail_points))))
        cv2.line(frame,
                 trail_points[i - 1],
                 trail_points[i],
                 (0, 255, 255),
                 thickness)

    # ---- Create Heatmap Overlay ----
    heatmap_display = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    heatmap_display = heatmap_display.astype(np.uint8)
    heatmap_display = cv2.applyColorMap(heatmap_display, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(frame, 0.7, heatmap_display, 0.3, 0)

    # ---- Display Speed ----
    cv2.putText(overlay,
                f"Speed: {speed_kmh:.1f} km/h",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2)

    # Write output
    out.write(overlay)

    cv2.imshow("YOLO Football Analytics", overlay)

    key = cv2.waitKey(10) & 0xFF
    if key == ord("q") or key == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()