# Dockerfile — containerized GraphRAG Compliance Search Engine
# Run:  docker build -t graphrag-engine .
#       docker run --rm -p 8501:8501 --env-file .env graphrag-engine
# Keys are supplied at runtime via --env-file (or -e) — never baked into the image.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Dependencies first: this layer caches unless requirements.txt changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code and the bundled demo corpora
COPY *.py ./
COPY *.pdf ./
COPY README.md sample.env ./

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501', timeout=5)" || exit 1

# Streamlit must bind 0.0.0.0 to be reachable outside the container
CMD ["streamlit", "run", "dashboard.py", "--server.port", "8501", "--server.address", "0.0.0.0", "--server.headless", "true"]