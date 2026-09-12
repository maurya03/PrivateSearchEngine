from __future__ import annotations

from dataclasses import dataclass
from .engine import SearchEngine

@dataclass(frozen=True)
class QueryCase:
    query: str
    relevant_urls: frozenset[str]


def precision_at_k(results: list[dict], relevant: set[str], k: int) -> float:
    shown = results[:k]
    return sum(r["url"] in relevant for r in shown) / max(1, len(shown))


def recall_at_k(results: list[dict], relevant: set[str], k: int) -> float:
    if not relevant: return 0.0
    return sum(r["url"] in relevant for r in results[:k]) / len(relevant)


def mrr(results: list[dict], relevant: set[str]) -> float:
    for i, result in enumerate(results, 1):
        if result["url"] in relevant: return 1.0 / i
    return 0.0


def evaluate(engine: SearchEngine, cases: list[QueryCase], k: int = 5) -> dict[str, float]:
    if not cases: return {"precision@k": 0.0, "recall@k": 0.0, "mrr": 0.0}
    values = [(precision_at_k(r := engine.search(c.query, k), set(c.relevant_urls), k), recall_at_k(r, set(c.relevant_urls), k), mrr(r, set(c.relevant_urls))) for c in cases]
    return {"precision@k": sum(v[0] for v in values)/len(values), "recall@k": sum(v[1] for v in values)/len(values), "mrr": sum(v[2] for v in values)/len(values)}
