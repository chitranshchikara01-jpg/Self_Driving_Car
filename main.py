import cv2
from lane_detection import detect_lane
from object_detection import detect_objects
from decision import make_decision

video_path = "video4.mp4"

cap = cv2.VideoCapture(video_path)

# Resizable window
cv2.namedWindow("Output", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Maintain aspect ratio (NO distortion)
    height, width = frame.shape[:2]

    screen_width = 1000
    screen_height = 700

    scale_w = screen_width / width
    scale_h = screen_height / height

    scale = min(scale_w, scale_h)

    new_w = int(width * scale)
    new_h = int(height * scale)

    frame = cv2.resize(frame, (new_w, new_h))

    # Lane Detection
    frame = detect_lane(frame)

    # Object Detection
    frame, objects = detect_objects(frame)

    # Decision
    decision = make_decision(objects)

    # Show decision
    cv2.putText(frame, f"Decision: {decision}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 255), 2)

    cv2.imshow("Output", frame)

    # ESC to exit
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
