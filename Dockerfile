FROM python:3.12-slim

WORKDIR /app

# System deps (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
  && rm -rf /var/lib/apt/lists/*

# Copy project
COPY pyproject.toml /app/
COPY src /app/src
COPY configs /app/configs
COPY artifacts /app/artifacts

# Install
RUN pip install --no-cache-dir -U pip \
 && pip install --no-cache-dir -e .

ENV LOG_FORMAT=json
ENV LOG_LEVEL=INFO
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "housing_model.service:app", "--host", "0.0.0.0", "--port", "8000"]

