FROM python:3

FROM python:3.10-slim-buster
WORKDIR /app

COPY . /app

RUN apt-get update && apt-get -y install cmake protobuf-compiler libzbar0

RUN apt-get update && apt-get -y install cmake protobuf-compiler libzbar0 ffmpeg libsm6 libxext6
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt



CMD [ "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000","--workers","2" ]