# Base image
FROM python:3.8-slim

WORKDIR /usr/src/mlops

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*




COPY requirements.txt ./requirements.txt
COPY setup.py ./setup.py
COPY src/ ./src/
ADD data.dvc ./data.dvc
ADD models.dvc ./models.dvc
ADD .dvc/ ./.dvc/
ADD .git/ ./.git/

RUN python -m pip install --upgrade pip
#necessary to install torch before torch-scatter, torch-sparse... workaround...
RUN pip install torch
RUN pip install -r requirements.txt --no-cache-dir

RUN dvc pull
