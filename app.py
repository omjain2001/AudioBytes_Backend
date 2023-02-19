from flask import Flask, request, redirect
import whisper
from tempfile import NamedTemporaryFile
from flask_cors import CORS
import re
import spacy
from fastdtw import fastdtw
import librosa
from scipy.spatial.distance import euclidean
import json


# Load the Whisper model:
model = whisper.load_model('base.en')

# spacy : to get context for a sentence
nlp = spacy.load("en_core_web_md")


app = Flask(__name__)
CORS(app)
app.secret_key = "caircocoders-ednalan"

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

                # get language
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
        result = re.findall('\\b'+search_word+'\\b',
                            i['text'], flags=re.IGNORECASE)
        if len(result) > 0:
            start_time = round(i['start'], 2)
            end_time = round(i['end'], 2)
            timestamps.append([start_time, end_time])

    return timestamps


@app.route('/getContextForSentence', methods=['POST'])
def getContextForSentence():

    content_type = request.headers.get('Content-Type')
    if (content_type == 'application/json'):
        req_data = request.get_json()
    else:
        return 'Content-Type not supported!'

    input_sentence = req_data["sentence"]
    text_array = req_data["transcripts"]

    doc1 = nlp(input_sentence)

    similarity_array_timestamp = []
    threshold = 0.5

    for segment in text_array:

        start_end = []
        doc2 = nlp(segment["text"])

        similarity_score = doc1.similarity(doc2)

        if (similarity_score >= threshold):
            start_end.append(segment["start"])
            start_end.append(segment["end"])

            similarity_array_timestamp.append(start_end)

    return similarity_array_timestamp


@app.route("/getTimestampsFromAudio", methods=["POST"])
def getTimestampsFromAudio():
    sampleAudioFile = request.files["inputAudio"]
    testAudioFile = request.files["sampleAudio"]

    test_audio, test_sr = librosa.load(testAudioFile)
    sample_audio, sample_sr = librosa.load(sampleAudioFile)

    # Display Test Audio Attributes
    print("Sampling rate: ", test_sr)
    print("Samples: ", test_audio)
    print("Frames Length: ", len(test_audio))
    print("Length of audio: ", (len(test_audio) / test_sr))
    print(f"Duration (sec): {librosa.get_duration(test_audio, sr=test_sr)}")

    # Display Test Audio Attributes
    print("Sampling rate: ", sample_sr)
    print("Samples: ", sample_audio)
    print("Frames Length: ", len(sample_audio))
    print("Length of audio: ", (len(sample_audio) / sample_sr))
    print(
        f"Duration (sec): {librosa.get_duration(sample_audio, sr=sample_sr)}")

    target_sr = 44100

    test_audio_resampled = librosa.resample(test_audio, test_sr, target_sr)
    test_audio_resampled = librosa.to_mono(test_audio_resampled)

    sample_audio_resampled = librosa.resample(
        sample_audio, sample_sr, target_sr)
    sample_audio_resampled = librosa.to_mono(sample_audio_resampled)

    window_size = 1024
    hop_length = 512
    test_mfccs = librosa.feature.mfcc(
        test_audio_resampled, target_sr, n_mfcc=20, hop_length=hop_length, n_fft=window_size)
    sample_mfccs = librosa.feature.mfcc(
        sample_audio_resampled, target_sr, n_mfcc=20, hop_length=hop_length, n_fft=window_size)

    print("Sample mfccs : ", sample_mfccs.shape)
    print("Test mfccs : ", test_mfccs.shape)

    sample_len = len(sample_audio_resampled)
    test_len = len(test_audio_resampled)

    slice_size = test_len
    step_size = test_len // 2

    # Set the threshold for considering a match
    threshold = 50

    # Initialize a list to store the times of the matches
    times = []
    distances = []
    output = {}

    for i in range(0, sample_len, step_size):
        if (i+slice_size > sample_len):
            break
        sliced_audio = sample_audio_resampled[i:i + slice_size]
        sliced_mfccs = librosa.feature.mfcc(
            sliced_audio, target_sr, n_mfcc=20, hop_length=hop_length, n_fft=window_size)

        # Calculate the DTW distance between the MFCCs
        sliced_mfccs = sliced_mfccs.flatten()
        test_mfccs = test_mfccs.flatten()
        distance, path = fastdtw(sliced_mfccs.reshape(
            -1, sliced_mfccs.shape[0]), test_mfccs.reshape(-1, test_mfccs.shape[0]), dist=euclidean)
        distances.append(distance)

        # Check if the DTW distance is below the threshold
        # if distance > threshold and distance > 0:
        if distance > 0:
            # Calculate the time of the match
            time = (i) / target_sr
            times.append(time)
            output[round(distance, 2)] = time

    sorted_dict = dict(sorted(output.items(), key=lambda x: x[0]))
    out_dict = dict(list(sorted_dict.items())[0: 10])

    for key, value in out_dict.items():
        print(round(key, 2), " : ", round(value, 2))

    return json.dumps(dict({"timestamps": list(out_dict.values())}))


if __name__ == '__main__':
    app.run(debug=True)
