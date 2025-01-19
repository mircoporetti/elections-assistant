FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml poetry.lock ./

RUN apt-get update \
    && apt-get install -y build-essential libpq-dev curl \
    && apt-get clean \
    && pip install poetry \
    && poetry install --no-root

COPY src/ ./src

EXPOSE 8000

CMD ["poetry", "run", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
