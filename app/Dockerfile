FROM python:3.7-slim

WORKDIR /app
COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

#COPY train.py train.py
#COPY jigsaw.csv jigsaw.csv

#CMD bash

CMD export GIT_PYTHON_REFRESH=quiet && python3 train.py
