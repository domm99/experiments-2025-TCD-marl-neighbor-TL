FROM python:3.12
RUN mkdir -p /experiment
#VOLUME "davide.domini-volume"
RUN pip install poetry==2.1.3
ENV DATA_DIR=/data
ENV POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1
WORKDIR /experiment
#COPY pyproject.toml poetry.lock* /experiment/
COPY . /experiment
RUN poetry install
ENV OWNER=1000:1000
CMD poetry run python3 src/main.py && \
    chown -R $OWNER $DATA_DIR