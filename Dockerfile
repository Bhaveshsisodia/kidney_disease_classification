# FROM python:3.9-slim-buster

# RUN apt update -y && apt install awscli -y
# WORKDIR /app

# COPY . /app
# RUN pip install -r requirements.txt

# CMD ["python", "app.py"]


# for AZURE

FROM python:3.9-slim-buster

WORKDIR /app
COPY . /app

RUN apt update -y
RUN pip install -r requirements.txt

CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]