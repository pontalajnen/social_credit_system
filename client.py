import cv2
import time
import requests


def main():
    url = "http://localhost:5000/upload"
    camera = cv2.VideoCapture(0)

    while True:
        ret, frame = camera.read()
        if not ret:
            print("Failed to capture image")
            break

        files = {"file": ("image.jpg", cv2.imencode(".jpg", frame)[1].tobytes())}

        time.sleep(1)

        try:
            response = requests.post(url, files=files)
            print(response)
        except requests.exceptions.RequestException as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
