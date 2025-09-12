FROM python:3.12
RUN mkdir -p /experiment
VOLUME "davide.domini-volume"
ENV DATA_DIR=/data
WORKDIR /experiment
COPY requirements.txt /experiment
RUN python3 -m pip install -r requirements.txt
COPY . /experiment
ENV OWNER=1000:1000
CMD python3 src/spread/main.py && \
    chown -R $OWNER $DATA_DIR