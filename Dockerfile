FROM python:3.7.6-buster

WORKDIR /app

COPY ai-roadmap-new-09335e3c56d2.json /tmp
ENV GOOGLE_APPLICATION_CREDENTIALS=/tmp/ai-roadmap-new-09335e3c56d2.json

COPY requirements.txt /tmp
RUN pip3 install -r /tmp/requirements.txt

COPY entrypoint.sh /app
COPY train/train.py /app
COPY test/test.py /app
COPY roc_curve/roc_curve_creator.py /app
COPY confusion_matrix/confusion_matrix.py /app

ENTRYPOINT ["./entrypoint.sh"]