import cv2
import numpy as np
from ultralytics import YOLO

# Carga modelo (se descarga solo la primera vez)
model = YOLO("yolov8n.pt")  # rápido y ligero

# Define tus cajones manualmente (coordenadas)
# (x1, y1, x2, y2)
PARKING_SPOTS = [
    (50, 100, 200, 260),
    (220, 100, 370, 260),
    (390, 100, 540, 260),
    (560, 100, 710, 260),
    # agrega más...
]


def detect_cars(frame):
    results = model(frame)[0]

    cars = []

    for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
        label = int(cls)

        # clase 2 = car en COCO
        if label == 2:
            x1, y1, x2, y2 = map(int, box)
            cars.append((x1, y1, x2, y2))

    return cars


def box_intersect(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    return not (ax2 < bx1 or ax1 > bx2 or ay2 < by1 or ay1 > by2)


def analyze_frame(frame):
    cars = detect_cars(frame)

    spots_status = []

    for spot in PARKING_SPOTS:
        occupied = False

        for car in cars:
            if box_intersect(spot, car):
                occupied = True
                break

        spots_status.append(occupied)

    return spots_status