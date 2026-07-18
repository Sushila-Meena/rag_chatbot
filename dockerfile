# use a slim python base — keeps the image small
FROM python:3.12-slim

# set working directory inside the container
WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# copy the rest of the project
COPY . .

# render sets the PORT env var automatically — uvicorn binds to it
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
