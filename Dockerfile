FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=300 \
    PIP_RETRIES=8 \
    HF_HOME=/data/model-cache

ARG PIP_INDEX_URL=https://mirrors.cloud.tencent.com/pypi/simple
ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

RUN groupadd --system anchor && useradd --system --gid anchor --home /app anchor
WORKDIR /app

COPY requirements.txt ./
RUN pip install --index-url "${PYTORCH_INDEX_URL}" torch && \
    pip install --index-url "${PIP_INDEX_URL}" --requirement requirements.txt

COPY . ./
RUN mkdir -p /data/anchor /data/model-cache && chown -R anchor:anchor /app /data

USER anchor
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=3)"

CMD ["python", "anchor_http.py"]
