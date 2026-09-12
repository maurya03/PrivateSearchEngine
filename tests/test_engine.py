from privatesearch.engine import SearchEngine, tokenize
from privatesearch.evaluation import QueryCase, evaluate

def seed(e):
    e.add("doc:a", "Database internals", "B+Tree indexes and write ahead logging make storage durable")
    e.add("doc:b", "Cooking", "Pasta and tomatoes make a simple dinner")
    e.add("doc:c", "Database recovery", "Crash recovery replays committed WAL records")

def test_bm25_and_field_boost(tmp_path):
    with SearchEngine(tmp_path / "s.db") as e:
        seed(e); r=e.search("database storage", 2); assert r[0]["url"] == "doc:a"

def test_reindex_is_idempotent(tmp_path):
    with SearchEngine(tmp_path / "s.db") as e:
        assert e.add("a", "Old", "old text") == e.add("a", "Old", "old text")
        e.add("a", "New", "new text"); assert e.search("new")[0]["title"] == "New"; assert e.count() == 1

def test_phrase_query(tmp_path):
    with SearchEngine(tmp_path / "s.db") as e:
        seed(e); assert e.search('"write ahead logging"')[0]["url"] == "doc:a"

def test_negative_term(tmp_path):
    with SearchEngine(tmp_path / "s.db") as e:
        seed(e); r=e.search("database -recovery"); assert r and all(x["url"] != "doc:c" for x in r)

def test_html_ingestion(tmp_path):
    docs=tmp_path/"docs"; docs.mkdir(); (docs/"x.html").write_text("<html><title>WAL</title><script>x</script><p>write ahead logging</p></html>")
    with SearchEngine(tmp_path/"s.db") as e:
        assert e.index_path(docs)==1; assert e.search("logging")[0]["title"] == "WAL"

def test_evaluation_metrics(tmp_path):
    with SearchEngine(tmp_path / "s.db") as e:
        seed(e); m=evaluate(e,[QueryCase("storage", frozenset({"doc:a"})), QueryCase("recovery", frozenset({"doc:c"}))]); assert m["mrr"] == 1.0

def test_tokenizer():
    assert tokenize("The WAL, storage and B+Tree!") == ["wal", "storage", "btree"]
