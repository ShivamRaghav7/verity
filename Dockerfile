FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY editguard ./editguard
COPY data/catalogue.json ./data/catalogue.json
RUN useradd --create-home editguard && mkdir runtime && chown editguard:editguard runtime
USER editguard
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "editguard.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
