FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml poetry.lock ./

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libpq-dev curl \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir poetry \
    && poetry config virtualenvs.create false \
    && poetry install --no-root --no-cache --without dev \
    && rm -rf /root/.cache

# Bake the Hugging Face models into the image so pods don't download them on every start
ENV HF_HOME=/app/.hf-cache
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('intfloat/multilingual-e5-small'); \
CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')"

COPY src/ ./src
COPY resources/manifests/ /app/resources/manifests/
RUN mkdir -p /app/faiss

ENV PYTHONPATH=/app/src


EXPOSE 8000

CMD ["poetry", "run", "uvicorn", "src.webapp.main:app", "--host", "0.0.0.0", "--port", "8000"]