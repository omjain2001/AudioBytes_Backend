from flask import Flask, json, request, jsonify
import os
import urllib.request
from werkzeug.utils import secure_filename
import whisper
import numpy as np
import io
import soundfile as sf
import librosa as lb
import miniaudio
from pydub import AudioSegment

app = Flask(__name__)
app.secret_key = "caircocoders-ednalan"

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['mp3', 'mpeg', 'wav'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/upload', methods=['POST'])
def upload_file():
    # check if the post request has the file part
    if 'file' not in request.files:
        resp = jsonify({'message': 'No file part in the request'})
        resp.status_code = 400
        return resp

    file = request.files['file']
    errors = {}
    success = True
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(
            app.config['UPLOAD_FOLDER'], filename).replace("\\", "/"))
        file_address = os.path.join(
            app.config['UPLOAD_FOLDER'], filename).replace("/", "\\")

        print(file_address)

        # converting auido to text using whisper
        # data, sample_rate = lb.load(io.BytesIO(
        #     file.read()), dtype="float32")
        # data = data.T
        # data_resampled = lb.resample(data, sample_rate, 44100)

        # data = AudioSegment.from_raw(io.BytesIO(file.read()))
        # print(data)

        model = whisper.load_model("base")
        transcript = model.transcribe(
            ".\\" + file_address, fp16=False)
        success = True
        print(transcript)

    else:
        errors[file.filename] = 'File type is not allowed'

    if success and errors:
        errors['message'] = 'File(s) successfully uploaded'
        resp = jsonify(errors)
        resp.status_code = 500
        return resp

    if success:
        # resp = jsonify({'message' : 'Files successfully uploaded'})
        resp = jsonify(transcript)
        resp.status_code = 201
        return resp
    else:
        resp = jsonify(errors)
        resp.status_code = 500
        return resp


if __name__ == '__main__':
    app.run(debug=True)
