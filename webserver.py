from flask import Flask, render_template
from flask_cors import CORS


app = Flask(__name__, static_folder="website", template_folder="website")

CORS(app)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/check_person", methods=["POST"])
def check_person():
    return {"status": "success", "message": "Person checked successfully"}


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
