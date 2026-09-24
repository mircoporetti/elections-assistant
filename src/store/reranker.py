import logging
import os

logger = logging.getLogger("uvicorn")

RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "12"))
RERANKING_ENABLED = os.getenv("RERANKING_ENABLED", "true").lower() != "false"

_cross_encoder = None


def is_enabled():
    return RERANKING_ENABLED


def load():
    global _cross_encoder
    if not RERANKING_ENABLED:
        logger.info("Reranking is disabled, skipping reranker model load.")
        return
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder

        logger.info("Loading reranker model %s...", RERANKER_MODEL)
        _cross_encoder = CrossEncoder(RERANKER_MODEL)
        logger.info("Reranker model has been loaded.")


def rerank(query, docs, k):
    if not RERANKING_ENABLED or not docs:
        return docs[:k]

    load()
    scores = _cross_encoder.predict([(query, doc.page_content) for doc in docs])
    ranked = sorted(zip(scores, docs), key=lambda scored: scored[0], reverse=True)
    return [doc for _, doc in ranked[:k]]


def rerank_with_scores(query, docs, k):
    if not RERANKING_ENABLED or not docs:
        return [(doc, None) for doc in docs[:k]]

    load()
    scores = _cross_encoder.predict([(query, doc.page_content) for doc in docs])
    ranked = sorted(zip(scores, docs), key=lambda scored: scored[0], reverse=True)
    return [(doc, float(score)) for score, doc in ranked[:k]]