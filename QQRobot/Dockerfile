FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl libxml2-dev libxslt-dev gcc && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN useradd --create-home --user-group appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 5000

CMD ["waitress-serve", "--host=0.0.0.0", "--port=5000", "main:app"]
