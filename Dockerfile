FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY hr_policy_navigator ./hr_policy_navigator
COPY streamlit_app.py ./
COPY sourcegraph_files ./sourcegraph_files
COPY certs ./certs
COPY evaluation_set.json evaluation_set_adversarial.json ./
RUN useradd -m app && mkdir -p traces state .cache && chown -R app /app
USER app
EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8501/_stcore/health')"
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
