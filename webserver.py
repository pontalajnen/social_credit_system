from flask import Flask, render_template, request  # type: ignore
from flask_cors import CORS  # type: ignore
import threading
import queue
import io
# import time
import os


# Create the app instance
app = Flask(__name__, static_folder="website", template_folder="website")
CORS(app)

# Queue between threads
image_queue = queue.Queue()

COLLECT_TRAINING_DATA = True
FILE_COUNTER = len(os.listdir("./uploads"))


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/check_person", methods=["POST"])
def check_person():
    return {"status": "success", "message": "Person checked successfully"}


@app.route("/upload", methods=["POST"])
def upload_file():
    print("here")
    if "file" not in request.files:
        return "No file part", 400

    file = request.files["file"]

    if file.filename == "":
        return "No selected file", 400

    if COLLECT_TRAINING_DATA:
        file.save(f"./uploads/{file.filename[0:-4]}-{FILE_COUNTER}.jpg")
        FILE_COUNTER += 1  # noqa

    image_queue.put(io.BytesIO(file.read()))

    return "File uploaded successfully", 200


def run_server():
    app.run(debug=False, host="0.0.0.0", port=5000)


def main():
    print("Webserver started!")

    while True:
        file = image_queue.get()
        print("Image received")

        file.seek(0)

        file.close()


if __name__ == "__main__":
    t1 = threading.Thread(target=run_server)
    t2 = threading.Thread(target=main)
    t1.start()
    t2.start()
