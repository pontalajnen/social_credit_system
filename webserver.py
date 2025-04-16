from flask import Flask, render_template, request
from flask_cors import CORS
import threading


app = Flask(__name__, static_folder="website", template_folder="website")

CORS(app)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/check_person", methods=["POST"])
def check_person():
    return {"status": "success", "message": "Person checked successfully"}


@app.route("/upload", methods=["GET"])
def upload_file():
    if "file" not in request.files:
        return "No file part", 400

    file = request.files["file"]

    if file.filename == "":
        return "No selected file", 400

    file.save(f"./uploads/{file.filename}")

    return "File uploaded successfully", 200


def run_server():
    app.run(debug=False, host="0.0.0.0", port=5000)


def main():
    print("Webserver started!")


if __name__ == "__main__":
    t1 = threading.Thread(target=run_server)
    t2 = threading.Thread(target=main)
    t1.start()
    t2.start()
