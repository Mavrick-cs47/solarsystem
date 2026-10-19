# Build context: ./backend   (docker compose handles this; see docker-compose.yml)
FROM python:3.12-slim

# Set to "true" to include TensorFlow for training/serving the image classifier (large image).
ARG INSTALL_VISION=false

WORKDIR /srv
COPY requirements.txt requirements-vision.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && if [ "$INSTALL_VISION" = "true" ]; then pip install --no-cache-dir -r requirements-vision.txt; fi

COPY app ./app
ENV MODEL_DIR=/srv/models
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
