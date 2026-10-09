FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
RUN pip install ".[api,aws,qdrant,mysql]"
COPY data ./data
RUN useradd --uid 10001 --no-create-home app && mkdir -p /app/var && chown app /app/var
USER 10001
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8080/healthz')"
CMD ["uvicorn", "support_assistant.api.app:app_factory", "--factory", "--host", "0.0.0.0", "--port", "8080"]
