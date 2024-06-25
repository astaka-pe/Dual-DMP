FROM python:3.10

WORKDIR /work
COPY . /work

RUN apt update
RUN apt install -y libgl1-mesa-dev

RUN pip install --upgrade pip
RUN pip install -r requirements.txt