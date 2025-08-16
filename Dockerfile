FROM python:3.11-slim

# 1) base setup
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2) copy source
COPY serving_app ./serving_app
COPY training    ./training

# 3) model artifact location (inside the image)
ENV MODEL_PATH=/app/models/model.pkl
RUN mkdir -p /app/models

# 4) train during build so the model is baked into the image
RUN python -m training.train

# 5) runtime config
ENV PORT=8000
EXPOSE 8000

# 6) single CMD that respects $PORT
CMD ["sh", "-c", "uvicorn serving_app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]

