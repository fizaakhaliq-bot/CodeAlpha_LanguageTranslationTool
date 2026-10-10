import cv2
import numpy as np
import time

def run_object_detection_and_tracking():
    """
    Real-time Object Detection and Color-based Object Tracking Pipeline.
    Uses OpenCV for video capture, frame processing, bounding box rendering, and centroid tracking.
    """
    print("=" * 60)
    print("  CODEALPHA AI INTERNSHIP - TASK 4: OBJECT DETECTION & TRACKING ")
    print("=" * 60)
    print("Initializing Webcam Feed...")
    
    # Initialize webcam capture (0 is default built-in/USB camera)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam or video source.")
        print("Creating synthetic video feed simulation mode...")
        # Fallback to demo mode if webcam is not accessible
        frame_width, frame_height = 640, 480
    else:
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Lower and upper bounds for detecting colored objects in HSV color space (e.g., Green/Blue)
    lower_color = np.array([35, 50, 50])
    upper_color = np.array([85, 255, 255])

    # Tracking storage
    tracking_history = []
    object_id = 1
    prev_time = time.time()

    print("Press 'q' in the video window to exit.")

    while True:
        if cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame from webcam. Exiting...")
                break
        else:
            # Generate animated test canvas if camera is unavailable
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            t = time.time()
            cx, cy = int(320 + 150 * np.cos(t)), int(240 + 100 * np.sin(t))
            cv2.circle(frame, (cx, cy), 30, (0, 255, 0), -1)

        # Convert RGB/BGR frame to HSV color space
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create binary mask for target color range
        mask = cv2.inRange(hsv_frame, lower_color, upper_color)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            # Find the largest contour in frame
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)

            if area > 500:  # Minimum pixel area threshold
                # Compute bounding box
                x, y, w, h = cv2.boundingRect(largest_contour)
                center_x = int(x + w / 2)
                center_y = int(y + h / 2)

                # Append center coordinates for path tracking
                tracking_history.append((center_x, center_y))
                if len(tracking_history) > 30:
                    tracking_history.pop(0)

                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Draw label background
                cv2.rectangle(frame, (x, y - 25), (x + 180, y), (0, 255, 0), -1)
                
                # Draw tracking ID and class label text
                label = f"ID #{object_id}: Object"
                cv2.putText(frame, label, (x + 5, y - 7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

                # Draw tracking centroid point
                cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

                # Draw motion trail path
                for i in range(1, len(tracking_history)):
                    if tracking_history[i - 1] is None or tracking_history[i] is None:
                        continue
                    thickness = int(np.sqrt(30 / float(i + 1)) * 2.5)
                    cv2.line(frame, tracking_history[i - 1], tracking_history[i], (0, 255, 255), thickness)

        current_time = time.time()
        fps = 1.0 / (current_time - prev_time) if (current_time - prev_time) > 0 else 30
        prev_time = current_time

        # Display FPS overlay
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "CodeAlpha AI Task 4 - Object Tracker", (10, frame_height - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Render display window
        cv2.imshow("CodeAlpha - Object Detection & Tracking", frame)

        # Exit loop on pressing 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up video capture resources
    if cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()
    print("Execution completed gracefully.")

if __name__ == "__main__":
    run_object_detection_and_tracking()
