from fastapi import FastAPI
import cv2
from app.detector import analyze_frame

app = FastAPI()

camera = cv2.VideoCapture(0)  # webcam
# o IP cam:
# camera = cv2.VideoCapture("rtsp://192.168.1.50:554/stream")


@app.get("/parking/status")
def parking_status():
    ret, frame = camera.read()

    if not ret:
        return {"error": "camera not available"}

    status = analyze_frame(frame)

    total = len(status)
    occupied = sum(status)
    free = total - occupied

    return {
        "total": total,
        "occupied": occupied,
        "free": free,
        "spots": status
    }