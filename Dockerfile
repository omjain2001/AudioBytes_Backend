FROM python:3.7-alpine3.17

WORKDIR /app/

RUN apk update && apk add git

COPY . .

# RUN pip install torch==1.13.1 -f https://download.pytorch.org/whl/torch_stable.html

RUN pip install flask 
# git+https://github.com/openai/whisper.git soundfile SpeechRecognition Flask-Cors

# CMD ["py", "main.py"]