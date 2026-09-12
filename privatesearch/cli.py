from __future__ import annotations
import argparse, json, time
from pathlib import Path
from .engine import SearchEngine
from .evaluation import QueryCase, evaluate

def main() -> None:
    p = argparse.ArgumentParser(prog="privatesearch")
    sub = p.add_subparsers(dest="cmd", required=True)
    ix = sub.add_parser("index"); ix.add_argument("path")
    s = sub.add_parser("search"); s.add_argument("query"); s.add_argument("-k", type=int, default=10); s.add_argument("--db", default="search.db"); s.add_argument("--field", choices=["all","title","body"], default="all")
    b = sub.add_parser("benchmark"); b.add_argument("query"); b.add_argument("--db", default="search.db"); b.add_argument("-n", type=int, default=100)
    e = sub.add_parser("evaluate"); e.add_argument("dataset"); e.add_argument("--db", default="search.db"); e.add_argument("-k", type=int, default=5)
    a = p.parse_args()
    if a.cmd == "index":
        with SearchEngine() as engine: print(f"indexed {engine.index_path(a.path)} documents; total={engine.count()}")
    elif a.cmd == "search":
        with SearchEngine(a.db) as engine:
            for r in engine.search(a.query, a.k, field=a.field): print(f"{r['score']:.4f}  {r['title']} — {r['url']}\n  {r['snippet']}")
    elif a.cmd == "benchmark":
        with SearchEngine(a.db) as engine:
            start=time.perf_counter()
            for _ in range(a.n): engine.search(a.query, 10)
            elapsed=time.perf_counter()-start
            print(json.dumps({"queries":a.n,"seconds":round(elapsed,6),"qps":round(a.n/elapsed,2),"avg_ms":round(elapsed/a.n*1000,3)}))
    else:
        data=json.loads(Path(a.dataset).read_text())
        cases=[QueryCase(x["query"], frozenset(x["relevant_urls"])) for x in data]
        with SearchEngine(a.db) as engine: print(json.dumps(evaluate(engine,cases,a.k), indent=2))

if __name__ == "__main__": main()
