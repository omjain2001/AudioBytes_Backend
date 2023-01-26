from flask import Flask, json, request, jsonify
import os
import urllib.request
from werkzeug.utils import secure_filename
import whisper
import numpy as np
import io
import soundfile as sf
from flask_cors import CORS


app = Flask(__name__)
CORS(app)
app.secret_key = "caircocoders-ednalan"

# UPLOAD_FOLDER = 'static/uploads'
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['mp3', 'mpeg', 'wav'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def main():
    return "Is this Working ?"

# input 
# data => data is the json object
# search_word => string word
@app.route('/timestamps', methods=['GET'])
def getTimestamps(data, search_word):
# def getTimestamps():
#     data = {
#   "language": "en",
#   "segments": [
#     {
#       "avg_logprob": -0.2746924082438151,
#       "compression_ratio": 1.120879120879121,
#       "end": 5.2,
#       "id": 0,
#       "no_speech_prob": 0.009703562594950199,
#       "seek": 0,
#       "start": 0.0,
#       "temperature": 0.0,
#       "text": " My dear Fanny, you feel these things are great deal too much.",
#       "tokens": [
#         1222, 6875, 479, 11612, 11, 291, 841, 613, 721, 366, 869, 2028, 886,
#         709, 13
#       ]
#     },
#     {
#       "avg_logprob": -0.4439869293799767,
#       "compression_ratio": 0.8695652173913043,
#       "end": 31.2,
#       "id": 1,
#       "no_speech_prob": 0.0006663693347945809,
#       "seek": 520,
#       "start": 5.2,
#       "temperature": 0.0,
#       "text": " I am most happy that you like the chain.",
#       "tokens": [
#         50364, 286, 669, 881, 2055, 300, 291, 411, 264, 5021, 13, 51664
#       ]
#     }
#   ],
#   "text": " My dear Fanny, you feel these things are great deal too much. I am most happy that you like the chain."
# }
#     search_word = "the"

    trans_segs = data["segments"]
    timestamps = []

    for i in trans_segs:
        if search_word in i['text']:
            start_time = round(i['start'],2)
            end_time = round(i['end'],2)
            timestamps.append([start_time, end_time])

    return timestamps
# usages :
# word = "the"
# timestamps = find_timestamps(data, word)

@app.route('/upload', methods=['POST'])

def upload_file():
    # check if the post request has the file part
    if 'file' not in request.files:
        resp = jsonify({'message': 'No file part in the request'})
        resp.status_code = 400
        return resp

    print("Entered upload file")
    file = request.files['file']
    errors = {}
    success = True
    if file and allowed_file(file.filename):
        # filename = secure_filename(file.filename)
        # file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        # file_address = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # converting auido to text using whisper
        data, sample_rate = sf.read(io.BytesIO(file.read()))
        model = whisper.load_model("base")
        transcript = model.transcribe(
            np.float32(data), fp16=False)
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
