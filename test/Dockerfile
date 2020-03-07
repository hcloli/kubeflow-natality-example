FROM google/cloud-sdk:latest

WORKDIR /app

COPY ai-roadmap-new-09335e3c56d2.json /tmp
ENV GOOGLE_APPLICATION_CREDENTIALS=/tmp/ai-roadmap-new-09335e3c56d2.json

COPY requirements.txt /tmp
RUN pip3 install -r /tmp/requirements.txt

COPY test.py /app

ENTRYPOINT ["python3", "test.py"]
