"""检索模块单测：BM25、向量编码、Faiss 四种索引、索引存取。"""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.bm25 import BM25Retriever  # noqa: E402
from rag.data import doc_text, load_corpus  # noqa: E402
from rag.dense import DenseRetriever, build_faiss_index, search_index  # noqa: E402
from rag.embedder import Embedder  # noqa: E402
from rag.text import ChineseTokenizer  # noqa: E402
from tests.fakes import DEMO_DIR, FakeSentenceTransformer, fake_embedder  # noqa: E402


def clustered_vectors(n=3000, d=64, centers=30, seed=0):
    rng = np.random.default_rng(seed)
    c = rng.normal(size=(centers, d))
    x = c[rng.integers(0, centers, n)] + 0.3 * rng.normal(size=(n, d))
    x = x.astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


class TestBM25(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        corpus = load_corpus(DEMO_DIR)
        cls.retriever = BM25Retriever(ChineseTokenizer("search"))
        cls.retriever.build([d["id"] for d in corpus], [doc_text(d) for d in corpus], show_progress=False)

    def test_keyword_query(self):
        hits = self.retriever.search(["宿舍里可以用电热毯吗"], top_k=5)[0]
        self.assertEqual(hits[0][0], "d14")
        self.assertTrue(all(s > 0 for _, s in hits))

    def test_oov_and_empty_queries(self):
        hits = self.retriever.search(["qwertyuiop", "！！！"], top_k=5)
        self.assertEqual(hits, [[], []])

    def test_top_k_larger_than_corpus(self):
        self.assertLessEqual(len(self.retriever.search(["学生"], top_k=1000)[0]), 30)

    def test_save_load(self):
        with tempfile.TemporaryDirectory() as d:
            self.retriever.save(d)
            loaded = BM25Retriever.load(d, ChineseTokenizer("search"))
            q = ["校园卡丢了怎么办"]
            self.assertEqual(loaded.search(q, 5), self.retriever.search(q, 5))
            with self.assertRaises(ValueError):
                BM25Retriever.load(d, ChineseTokenizer("accurate"))
            with self.assertRaises(ValueError):  # k1 改了但没重建索引
                BM25Retriever.load(d, ChineseTokenizer("search"), k1=0.5, b=0.75)
            ud = Path(d) / "dict.txt"
            ud.write_text("电热毯\n", encoding="utf-8")
            with self.assertRaises(ValueError):  # 加了领域词典但没重建索引
                BM25Retriever.load(d, ChineseTokenizer("search", user_dict=str(ud)))


class TestEmbedder(unittest.TestCase):
    def test_instruction_and_normalization(self):
        model = FakeSentenceTransformer()
        emb = Embedder(model, name="m", query_instruction="指令：")
        q = emb.encode_queries(["你好", "世界"])
        self.assertEqual(model.calls[-1], ["指令：你好", "指令：世界"])
        self.assertEqual(q.dtype, np.float32)
        np.testing.assert_allclose(np.linalg.norm(q, axis=1), 1.0, rtol=1e-5)
        emb.encode_docs(["文档"])
        self.assertEqual(model.calls[-1], ["文档"])  # 文档不加指令


class TestFaiss(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x = clustered_vectors()
        cls.q = cls.x[:50] + 0.05
        cls.q /= np.linalg.norm(cls.q, axis=1, keepdims=True)
        cls.ids = [f"d{i}" for i in range(len(cls.x))]
        exact = search_index(build_faiss_index(cls.x, {"index_type": "flat"}), cls.ids, cls.q, 10)
        cls.exact = [{d for d, _ in h} for h in exact]

    def recall(self, cfg):
        hits = search_index(build_faiss_index(self.x, cfg), self.ids, self.q, 10)
        return np.mean([len(e & {d for d, _ in h}) / 10 for e, h in zip(self.exact, hits)])

    def test_index_types(self):
        self.assertEqual(self.recall({"index_type": "flat"}), 1.0)
        self.assertGreaterEqual(self.recall({"index_type": "hnsw", "hnsw_m": 16, "hnsw_ef_search": 128}), 0.9)
        self.assertEqual(self.recall({"index_type": "ivf", "ivf_nlist": 16, "ivf_nprobe": 16}), 1.0)  # 搜全部簇 = 精确
        self.assertGreater(self.recall({"index_type": "ivf", "ivf_nlist": 16, "ivf_nprobe": 4}), 0.5)
        self.assertGreater(self.recall({"index_type": "ivfpq", "ivf_nlist": 16, "ivf_nprobe": 16, "pq_m": 16}), 0.3)

    def test_bad_params(self):
        with self.assertRaises(ValueError):
            build_faiss_index(self.x, {"index_type": "ivfpq", "pq_m": 7})
        with self.assertRaises(ValueError):
            build_faiss_index(self.x, {"index_type": "annoy"})

    def test_ivfpq_needs_enough_vectors(self):
        with self.assertRaises(ValueError):
            build_faiss_index(self.x[:100], {"index_type": "ivfpq", "pq_m": 16, "pq_nbits": 8})

    def test_small_corpus_nlist_is_clamped(self):
        index = build_faiss_index(self.x[:100], {"index_type": "ivf", "ivf_nlist": 1024, "ivf_nprobe": 64})
        self.assertEqual(index.nlist, 2)  # 100 // 39
        self.assertEqual(index.nprobe, 2)


class TestDenseRetriever(unittest.TestCase):
    def test_build_save_load_and_switch_index(self):
        corpus = load_corpus(DEMO_DIR)
        ids, texts = [d["id"] for d in corpus], [doc_text(d) for d in corpus]
        emb = fake_embedder()
        r = DenseRetriever(emb, {"index_type": "flat"})
        r.build(ids, texts, show_progress=False)
        hits = r.search(["宿舍里可以用电热毯吗"], 3)[0]
        self.assertEqual(hits[0][0], "d14")
        with tempfile.TemporaryDirectory() as d:
            r.save(d)
            loaded = DenseRetriever.load(d, emb, {"index_type": "flat"})
            self.assertEqual(loaded.search(["校园卡丢了"], 3), r.search(["校园卡丢了"], 3))
            hnsw = DenseRetriever.load(d, emb, {"index_type": "hnsw", "hnsw_m": 8})  # 用存好的向量现建
            self.assertTrue((Path(d) / "faiss_hnsw.index").exists())
            self.assertEqual(len(hnsw.search(["校园卡丢了"], 3)[0]), 3)
            with self.assertRaises(ValueError):
                DenseRetriever.load(d, fake_embedder("another-model"), {"index_type": "flat"})

    def test_stale_index_is_rebuilt(self):
        corpus = load_corpus(DEMO_DIR)
        ids, texts = [d["id"] for d in corpus], [doc_text(d) for d in corpus]
        emb = fake_embedder()
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp) / "中文目录"  # Windows 中文路径下 faiss.write_index 会失败，这里走序列化
            r = DenseRetriever(emb, {"index_type": "flat"})
            r.build(ids, texts, show_progress=False)
            r.save(d)
            self.assertEqual(DenseRetriever.load(d, emb, {"index_type": "hnsw"}).index.ntotal, 30)  # 缓存 hnsw
            r2 = DenseRetriever(emb, {"index_type": "flat"})
            r2.build(ids[:20], texts[:20], show_progress=False)  # 语料变了，重新编码
            r2.save(d)
            hnsw = DenseRetriever.load(d, emb, {"index_type": "hnsw"})
            self.assertEqual(hnsw.index.ntotal, 20)  # 旧的 30 条 hnsw 缓存被识别为过期并重建
            again = DenseRetriever.load(d, emb, {"index_type": "hnsw", "hnsw_ef_construction": 64})
            self.assertEqual(again.index.hnsw.efConstruction, 64)  # 建索引参数变了也会重建
            emb.model.max_seq_length = 256
            with self.assertRaises(ValueError):  # max_seq_length 变了，向量要重新编码
                DenseRetriever.load(d, emb, {"index_type": "flat"})


if __name__ == "__main__":
    unittest.main()
