from __future__ import annotations

import hashlib
import math
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from bs4 import BeautifulSoup

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+(?:\+[A-Za-z0-9_]+)*", re.UNICODE)
STOPWORDS = frozenset("the a an and or to of in on for with is are was were be this that from by as at it its into over under a about after before between during through without not".split())

@dataclass(frozen=True)
class SearchResult:
    id: int
    url: str
    title: str
    score: float
    snippet: str

    def as_dict(self) -> dict:
        return {"id": self.id, "url": self.url, "title": self.title, "score": round(self.score, 6), "snippet": self.snippet}


def tokenize(text: str, *, remove_stopwords: bool = True) -> list[str]:
    tokens = [t.lower().replace("+", "") for t in TOKEN_RE.findall(text)]
    return [t for t in tokens if not remove_stopwords or t not in STOPWORDS]


def _normalize_url(value: str) -> str:
    return value.strip()


class SearchEngine:
    """Local-first inverted-index search engine with BM25 ranking.

    SQLite stores the document catalog and postings. The implementation keeps
    indexing deterministic and inspectable so it can later serve as a retrieval
    component for RAG systems.
    """

    def __init__(self, db_path: str | Path = "search.db", *, k1: float = 1.5, b: float = 0.75):
        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("BM25 parameters require k1 > 0 and 0 <= b <= 1")
        self.db = sqlite3.connect(str(db_path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.k1, self.b = k1, b
        self._setup()

    def _setup(self) -> None:
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS documents(
          id INTEGER PRIMARY KEY, url TEXT UNIQUE NOT NULL, title TEXT NOT NULL,
          body TEXT NOT NULL, content_hash TEXT NOT NULL, token_count INTEGER NOT NULL,
          created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS postings(
          term TEXT NOT NULL, doc_id INTEGER NOT NULL, field TEXT NOT NULL,
          freq INTEGER NOT NULL, positions TEXT NOT NULL DEFAULT '',
          PRIMARY KEY(term, doc_id, field), FOREIGN KEY(doc_id) REFERENCES documents(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_postings_term ON postings(term);
        CREATE INDEX IF NOT EXISTS idx_postings_doc ON postings(doc_id);
        CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        """)
        self.db.commit()

    def close(self) -> None:
        self.db.close()

    def __enter__(self): return self
    def __exit__(self, *_): self.close()

    @staticmethod
    def _hash(title: str, body: str) -> str:
        return hashlib.sha256((title + "\0" + body).encode("utf-8")).hexdigest()

    @staticmethod
    def _postings(text: str) -> tuple[int, dict[str, tuple[int, str]]]:
        terms = tokenize(text)
        counts: dict[str, list[int]] = {}
        for pos, term in enumerate(terms): counts.setdefault(term, []).append(pos)
        return len(terms), {t: (len(pos), ",".join(map(str, pos))) for t, pos in counts.items()}

    def add(self, url: str, title: str, body: str) -> int:
        """Insert or update one document. Returns its stable document id."""
        url = _normalize_url(url)
        digest = self._hash(title, body)
        old = self.db.execute("SELECT id, content_hash FROM documents WHERE url=?", (url,)).fetchone()
        if old and old["content_hash"] == digest:
            return int(old["id"])
        if old:
            doc_id = int(old["id"])
            self.db.execute("UPDATE documents SET title=?,body=?,content_hash=?,token_count=?,updated_at=CURRENT_TIMESTAMP WHERE id=?", (title, body, digest, len(tokenize(body)), doc_id))
            self.db.execute("DELETE FROM postings WHERE doc_id=?", (doc_id,))
        else:
            doc_id = int(self.db.execute("INSERT INTO documents(url,title,body,content_hash,token_count) VALUES(?,?,?,?,?)", (url, title, body, digest, len(tokenize(body)))).lastrowid)
        rows = []
        for field, text, weight in (("title", title, 2.5), ("body", body, 1.0)):
            _, data = self._postings(text)
            rows.extend((term, doc_id, field, freq, positions) for term, (freq, positions) in data.items())
        self.db.executemany("INSERT INTO postings(term,doc_id,field,freq,positions) VALUES(?,?,?,?,?)", rows)
        self.db.commit()
        return doc_id

    def remove(self, url: str) -> bool:
        cur = self.db.execute("DELETE FROM documents WHERE url=?", (_normalize_url(url),))
        self.db.commit()
        return cur.rowcount > 0

    def index_path(self, root: str | Path) -> int:
        root = Path(root)
        count = 0
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in {".txt", ".md", ".html", ".htm"}: continue
            raw = path.read_text(encoding="utf-8", errors="ignore")
            title, body = self._parse_file(path, raw)
            self.add(str(path.resolve()), title, body); count += 1
        return count

    @staticmethod
    def _parse_file(path: Path, raw: str) -> tuple[str, str]:
        if path.suffix.lower() in {".html", ".htm"}:
            soup = BeautifulSoup(raw, "html.parser")
            title = soup.title.get_text(" ", strip=True) if soup.title else path.name
            for tag in soup(["script", "style", "noscript"]): tag.decompose()
            return title, soup.get_text(" ", strip=True)
        return path.name, raw

    def count(self) -> int:
        return int(self.db.execute("SELECT COUNT(*) FROM documents").fetchone()[0])

    def _doc_stats(self) -> tuple[int, float]:
        row = self.db.execute("SELECT COUNT(*) n, COALESCE(AVG(token_count),1) avg_len FROM documents").fetchone()
        return int(row["n"] or 0), float(row["avg_len"] or 1)

    def _candidate_terms(self, terms: Iterable[str]) -> set[int]:
        terms = list(dict.fromkeys(terms))
        if not terms: return set()
        placeholders = ",".join("?" * len(terms))
        return {int(r[0]) for r in self.db.execute(f"SELECT DISTINCT doc_id FROM postings WHERE term IN ({placeholders})", terms)}

    def _phrase_docs(self, phrase: str) -> set[int]:
        terms = tokenize(phrase)
        if not terms: return set()
        candidates = self._candidate_terms(terms)
        hits = set()
        for doc_id in candidates:
            positions_by_term = {}
            for term in terms:
                row = self.db.execute("SELECT positions FROM postings WHERE term=? AND doc_id=? AND field='body'", (term, doc_id)).fetchone()
                if not row: break
                positions_by_term[term] = {int(x) for x in row["positions"].split(",") if x}
            else:
                first = positions_by_term[terms[0]]
                if any(all((p + offset) in positions_by_term[t] for offset, t in enumerate(terms)) for p in first): hits.add(doc_id)
        return hits

    def search(self, query: str, k: int = 10, *, field: str = "all", phrase_boost: float = 1.5) -> list[dict]:
        if k <= 0: return []
        query = query.strip()
        phrases = re.findall(r'"([^\"]+)"', query)
        cleaned = re.sub(r'"[^\"]+"', ' ', query)
        terms = tokenize(cleaned)
        N, avgdl = self._doc_stats()
        if not N: return []

        # Simple AND/OR/NOT query operators. Unqualified terms use OR semantics.
        excluded = {x.lower() for x in re.findall(r"-(\w+)", cleaned)}
        positive = [t for t in terms if t not in excluded]
        candidates = self._candidate_terms(positive) if positive else set(range(1, N + 1))
        for phrase in phrases: candidates |= self._phrase_docs(phrase)
        for term in excluded: candidates -= self._candidate_terms([term])

        scores: dict[int, float] = {}
        field_filter = {"title", "body", "all"}
        if field not in field_filter: raise ValueError("field must be title, body, or all")
        for term in positive:
            df = int(self.db.execute("SELECT COUNT(DISTINCT doc_id) FROM postings WHERE term=?", (term,)).fetchone()[0])
            if not df: continue
            idf = math.log(1 + (N - df + 0.5) / (df + 0.5))
            rows = self.db.execute("SELECT p.doc_id,p.field,p.freq,d.token_count FROM postings p JOIN documents d ON d.id=p.doc_id WHERE p.term=?", (term,))
            for r in rows:
                if r["doc_id"] not in candidates or (field != "all" and r["field"] != field): continue
                tf = r["freq"] * (self.k1 + 1) / (r["freq"] + self.k1 * (1 - self.b + self.b * r["token_count"] / avgdl))
                boost = 2.5 if r["field"] == "title" else 1.0
                scores[int(r["doc_id"])] = scores.get(int(r["doc_id"]), 0.0) + idf * tf * boost
        for phrase in phrases:
            for doc_id in self._phrase_docs(phrase): scores[doc_id] = scores.get(doc_id, 0) + phrase_boost

        ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:k]
        results = []
        qterms = terms + [t for p in phrases for t in tokenize(p)]
        for doc_id, score in ranked:
            row = self.db.execute("SELECT id,url,title,body FROM documents WHERE id=?", (doc_id,)).fetchone()
            results.append(SearchResult(doc_id, row["url"], row["title"], score, self._snippet(row["body"], qterms)).as_dict())
        return results

    @staticmethod
    def _snippet(body: str, terms: list[str], width: int = 220) -> str:
        low = body.lower(); positions = [low.find(t.lower()) for t in terms if low.find(t.lower()) >= 0]
        pos = min(positions) if positions else 0
        start = max(0, pos - width // 3)
        end = min(len(body), start + width)
        snippet = " ".join(body[start:end].replace("\n", " ").split())
        return ("…" if start else "") + snippet + ("…" if end < len(body) else "")

    def all_documents(self) -> list[dict]:
        return [dict(r) for r in self.db.execute("SELECT id,url,title,token_count,created_at,updated_at FROM documents ORDER BY id")]
