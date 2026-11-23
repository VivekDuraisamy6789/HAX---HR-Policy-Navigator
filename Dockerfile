FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true

WORKDIR /app

# Install system dependencies for audio, offline TTS (espeak-ng), and healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
    espeak-ng \
    libespeak1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application source & configurations
COPY hr_policy_navigator ./hr_policy_navigator
COPY streamlit_app.py ./
COPY sourcegraph_files ./sourcegraph_files
COPY certs ./certs
COPY evaluation_set.json evaluation_set_adversarial.json ./
COPY .streamlit ./.streamlit

# Create runtime directories with non-root ownership
RUN useradd -m -u 1000 app && \
    mkdir -p traces state .cache && \
    chown -R app:app /app

USER app

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8501/_stcore/health')"

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
