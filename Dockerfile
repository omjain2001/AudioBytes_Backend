FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install git -y
RUN pip3 install "git+https://github.com/openai/whisper.git" 
RUN pip3 install flask flask_cors librosa fastdtw scipy
RUN pip3 install -U spacy
RUN pip3 install "https://github.com/explosion/spacy-models/releases/download/en_core_web_md-3.0.0/en_core_web_md-3.0.0.tar.gz"
RUN apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

COPY . .

EXPOSE 5000

CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]
