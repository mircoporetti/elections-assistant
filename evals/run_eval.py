import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from chat.party import Party
from store import reranker as reranker_module
from store import vector_store

GOLDEN_SET_PATH = os.path.join(os.path.dirname(__file__), "golden_set.json")


def chunk_id(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def rank_of_expected(docs, item):
    for position, doc in enumerate(docs, start=1):
        if chunk_id(doc.page_content) == item["expected_chunk_id"]:
            return position
    return None


def rank_of_expected_page(docs, item):
    for position, doc in enumerate(docs, start=1):
        same_source = doc.metadata.get("source") == item["expected_source"]
        if same_source and doc.metadata.get("page") == item["expected_page"]:
            return position
    return None


def evaluate(label, retrieve, golden_set, k):
    exact_hits = 0
    page_hits = 0
    reciprocal_ranks = 0.0
    party_correct = 0
    empty_results = 0

    for item in golden_set:
        detected = Party.get_from_message(item["question"])
        if detected and detected.name == item["expected_party"]:
            party_correct += 1

        docs = retrieve(item["question"], detected)
        if not docs:
            empty_results += 1

        rank = rank_of_expected(docs, item)
        if rank:
            exact_hits += 1
            reciprocal_ranks += 1.0 / rank
        if rank_of_expected_page(docs, item):
            page_hits += 1

    total = len(golden_set)
    print(f"{label}")
    print(f"  recall@{k} (exact chunk) : {exact_hits}/{total} = {100 * exact_hits / total:.1f}%")
    print(f"  recall@{k} (source+page) : {page_hits}/{total} = {100 * page_hits / total:.1f}%")
    print(f"  MRR@{k}                  : {reciprocal_ranks / total:.3f}")
    print(f"  party routing            : {party_correct}/{total} = {100 * party_correct / total:.1f}%")
    print(f"  empty result sets        : {empty_results}")
    print()
    return {"recall": exact_hits / total, "page_recall": page_hits / total, "mrr": reciprocal_ranks / total}


def dense_retrieve(question, party, k):
    store = vector_store.vector_store
    party_filter = vector_store.party_filter_for(party)
    return store.similarity_search(
        question, k=k, filter=party_filter, fetch_k=vector_store.fetch_k_for(party_filter)
    )


def reranked_retrieve(question, party, k, reranker, candidates=12):
    pool = dense_retrieve(question, party, candidates)
    if not pool:
        return []
    scores = reranker.predict([(question, doc.page_content) for doc in pool])
    ranked = [doc for _, doc in sorted(zip(scores, pool), key=lambda pair: pair[0], reverse=True)]
    return ranked[:k]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--candidates", type=int, default=12)
    parser.add_argument("--reranker", default=reranker_module.RERANKER_MODEL)
    parser.add_argument("--skip-reranker", action="store_true")
    args = parser.parse_args()

    with open(GOLDEN_SET_PATH, encoding="utf-8") as source:
        golden_set = json.load(source)

    vector_store.init()
    print(f"golden set: {len(golden_set)} questions, k={args.k}, candidate pool={args.candidates}")
    print(f"index: {vector_store.vector_store.index.ntotal} vectors, "
          f"{vector_store.vector_store.index.d} dims, model {vector_store.EMBEDDING_MODEL}")
    print()

    results = {}
    results["dense"] = evaluate(
        "A. dense only (no reranking)",
        lambda q, p: dense_retrieve(q, p, args.k), golden_set, args.k)

    if not args.skip_reranker:
        import time

        from sentence_transformers import CrossEncoder

        reranker = CrossEncoder(args.reranker)
        print(f"reranker: {args.reranker}, pool={args.candidates}")

        started = time.time()
        results["dense+rerank"] = evaluate(
            "B. dense + rerank (shipped)",
            lambda q, p: reranked_retrieve(q, p, args.k, reranker, args.candidates),
            golden_set, args.k)

        print(f"rerank cost: {(time.time() - started) / len(golden_set) * 1000:.0f} ms per query "
              f"({args.candidates} pairs scored) on this machine")
        print()

    print("summary (recall@%d / MRR@%d):" % (args.k, args.k))
    baseline = results["dense"]
    for label, scores in results.items():
        delta = scores["recall"] - baseline["recall"]
        print(f"  {label:16s} recall {100 * scores['recall']:5.1f}%  "
              f"MRR {scores['mrr']:.3f}  ({delta * 100:+.1f} pts vs dense)")


if __name__ == "__main__":
    main()