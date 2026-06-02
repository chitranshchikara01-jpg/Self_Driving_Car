import cv2
from lane_detection import detect_lane
from object_detection import detect_objects
from decision import make_decision

# Webcam open (0 = default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Camera not opening")
    exit()

cv2.namedWindow("Live Camera", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()

    if not ret:
        print("❌ Frame not received")
        break

    # Flip for natural view (optional)
    frame = cv2.flip(frame, 1)

    # Maintain aspect ratio
    height, width = frame.shape[:2]

    max_width = 900
    if width > max_width:
        scale = max_width / width
        frame = cv2.resize(frame, (int(width * scale), int(height * scale)))

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

    cv2.imshow("Live Camera", frame)

    # ESC press → exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()