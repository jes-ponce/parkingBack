import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

PARKING_SPOTS = [
    (50, 100, 200, 260),
    (220, 100, 370, 260),
    (390, 100, 540, 260),
    (560, 100, 710, 260),
]


def intersect(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (ax2 < bx1 or ax1 > bx2 or ay2 < by1 or ay1 > by2)


def run_live_camera():

    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)  # macOS fix

    if not cap.isOpened():
        print("❌ Camera not found at index 0, trying index 1...")
        cap = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            print("❌ No camera found. Check System Settings → Privacy & Security → Camera.")
            return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Warm-up
    for _ in range(5):
        cap.read()

    print("✅ Camera opened successfully. Press ESC to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame.")
            break

        results = model(frame)[0]

        cars = []

        for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
            if int(cls) == 2:  # car
                x1, y1, x2, y2 = map(int, box)
                cars.append((x1, y1, x2, y2))

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, "CAR", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        occupied_count = 0

        for spot in PARKING_SPOTS:
            occ = any(intersect(spot, car) for car in cars)

            color = (0, 0, 255) if occ else (0, 255, 0)

            if occ:
                occupied_count += 1

            x1, y1, x2, y2 = spot
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        total = len(PARKING_SPOTS)
        free = total - occupied_count

        cv2.putText(frame, f"Libre: {free}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.putText(frame, f"Ocupado: {occupied_count}", (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Parking Detector", frame)

        if cv2.waitKey(1) == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_live_camera()