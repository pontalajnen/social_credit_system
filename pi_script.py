import cv2
import time
import requests


def main():
    url = "http://localhost:5000/api/check_person"
    camera = cv2.VideoCapture(0)

    while True:
        ret, frame = camera.read()
        if not ret:
            print("Failed to capture image")
            break

        files = {"file": ("image.jpg", cv2.imencode(".jpg", frame)[1].tobytes())}

        time.sleep(1)


if __name__ == "__main__":
    main()
