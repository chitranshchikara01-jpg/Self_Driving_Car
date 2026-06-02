from ultralytics import YOLO
import cv2

# model load
model = YOLO("yolov8n.pt")

def detect_objects(frame):
    objects = []

    results = model(frame, conf=0.25)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # width of object
            box_width = x2 - x1

            # simple distance formula
            distance = int(1000 / (box_width + 1))
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = model.names[cls]

            objects.append((label, conf, distance))

            # draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # draw label
            cv2.putText(frame, f"{label} {distance}cm",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 0, 0), 2)

    return frame, objects