import cv2
import numpy as np
import time

# Open video file
cap = cv2.VideoCapture("match.mp4")

# Check if video opened
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

prev_center = None
prev_time = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Blur to reduce noise
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)

    # Convert to HSV
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # Define white color range
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 40, 255])

    mask = cv2.inRange(hsv, lower_white, upper_white)

    # Clean mask
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)

        if radius > 5:
            center = (int(x), int(y))

            # Draw circle
            cv2.circle(frame, center, int(radius), (0, 255, 0), 2)

            # Speed calculation
            current_time = time.time()

            if prev_center is not None and prev_time is not None:
                distance = np.sqrt(
                    (center[0] - prev_center[0])**2 +
                    (center[1] - prev_center[1])**2
                )
                time_diff = current_time - prev_time

                if time_diff > 0:
                    speed = distance / time_diff
                    cv2.putText(
                        frame,
                        f"Speed: {speed:.2f} px/s",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2
                    )

            prev_center = center
            prev_time = current_time

    cv2.imshow("Football Tracker", frame)

    key = cv2.waitKey(25) & 0xFF
    if key == ord("q") or key == 27:
        break

cap.release()
cv2.destroyAllWindows()