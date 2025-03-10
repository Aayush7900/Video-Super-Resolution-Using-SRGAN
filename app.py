import os
from flask import Flask, request, render_template, send_from_directory, redirect, url_for
from werkzeug.utils import secure_filename
from processing import enhance_video 

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
PROCESSED_FOLDER = 'static/processed'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "video" not in request.files:
            return "No file uploaded", 400

        video = request.files["video"]
        if video.filename == "":
            return "No selected file", 400

        filename = secure_filename(video.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        video.save(filepath)

        # Process video
        processed_path = os.path.join(app.config["PROCESSED_FOLDER"], filename)
        enhance_video(filepath, processed_path)  # Your function

        return redirect(url_for("download", filename=filename))

    return render_template("index.html")

@app.route("/download/<filename>")
def download(filename):
    return render_template("download.html", filename=filename)

@app.route("/processed/<filename>")
def processed_file(filename):
    return send_from_directory(app.config["PROCESSED_FOLDER"], filename)

if __name__ == "__main__":
    app.run(debug=True)
