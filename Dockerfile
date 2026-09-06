FROM python:3.12-slim
WORKDIR /app
COPY requirements.lock .
RUN pip install --no-cache-dir -r requirements.lock
COPY gero_stability gero_stability
COPY tests tests
COPY pyproject.toml .
ENTRYPOINT ["python", "-m", "gero_stability.cli"]
CMD ["run", "--output", "/reports", "--random-cases", "24"]
