FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY serving_app ./serving_app
COPY training ./training
RUN python -m training.train || true
ENV PORT=8000
EXPOSE 8000
CMD ["uvicorn","serving_app.main:app","--host","0.0.0.0","--port","8000"]
