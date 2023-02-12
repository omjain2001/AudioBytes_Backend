from flask import Flask, request, redirect
import whisper
from tempfile import NamedTemporaryFile
from flask_cors import CORS
import re

# Load the Whisper model:
model = whisper.load_model('medium')

app = Flask(__name__)
CORS(app)
app.secret_key = "caircocoders-ednalan"
CORS(app)

ALLOWED_EXTENSIONS = set(['mp3', 'mpeg', 'wav'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def main():
    return "Is this Working ?"


@app.route("/upload", methods=["POST"])
def upload():
    transcript = ""
    transcripts = {}
    if request.method == "POST":
        if "file" not in request.files:
            return "No File Uploaded"

        file = request.files["file"]
        if file.filename == "":
            return "Filename cant be empty"

        print("Form Data Received !!")
        if file:
            uploads_dict = request.files.to_dict()
            print("Items : ", uploads_dict.items())

            for fileName, fileStorage in uploads_dict.items():
                temp = NamedTemporaryFile()
                fileStorage.save(temp)
                result = model.transcribe(temp.name)
                transcripts = result


    return transcripts


def getLanguage(audio):
    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(audio)
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # detect the spoken language
    _, probs = model.detect_language(mel)
    lang = max(probs, key=probs.get)
    print(f"Detected language: {lang}")

    return lang

@app.route("/final", methods=["POST"])
def final():
    transcripts = {}
    if request.method == "POST":
        if "file" not in request.files:
            return "No File Uploaded"

        file = request.files["file"]
        if file.filename == "":
            return "Filename cant be empty"

        print("Form Data Received !!")
        if file:
            uploads_dict = request.files.to_dict()
            print("Items : ", uploads_dict.items())

            for fileName, fileStorage in uploads_dict.items():
                temp = NamedTemporaryFile()
                fileStorage.save(temp)

                #get language
                lang = getLanguage(temp.name)

                # get transcription
                result = model.transcribe(temp.name)
                transcripts = result
        
        return transcripts


@app.route('/timestamps', methods=['POST'])
def getTimestamps():

    content_type = request.headers.get('Content-Type')
    if (content_type == 'application/json'):
        req_data = request.get_json()
    else:
        return 'Content-Type not supported!'

    search_word = ""
    transcript_data = {}
    

    if req_data:
        if 'transcript_data' in req_data:
            transcript_data = req_data['transcript_data']

        if 'search_word' in req_data:
            search_word = req_data['search_word']
    
    trans_segs = transcript_data['segments']
    timestamps = []

    for i in trans_segs:
        result = re.findall('\\b'+search_word+'\\b', i['text'], flags=re.IGNORECASE)
        if len(result)>0:
            start_time = round(i['start'], 2)
            end_time = round(i['end'], 2)
            timestamps.append([start_time, end_time])
           

    return timestamps


if __name__ == '__main__':
    app.run(debug=True)
