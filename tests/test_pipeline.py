"""端到端冒烟测试：在 demo 数据上用假模型跑通 建索引 -> 消融评测 -> 失败分析 -> 延迟 -> 问答，以及命令行脚本。"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.evaluation import MODES, analyze_failures, format_table, measure_latency, run_eval, save_results  # noqa: E402
from rag.generator import Generator, build_messages  # noqa: E402
from rag.indexing import build_bm25, build_dense  # noqa: E402
from rag.pipeline import RAGPipeline, rerank_head  # noqa: E402
from rag.data import load_split_ids, write_jsonl, write_qrels  # noqa: E402
from rag.prep import make_splits_for, prepare_docs  # noqa: E402
from tests.fakes import ROOT, demo_config, fake_embedder, fake_reranker  # noqa: E402


class TestEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.cfg = demo_config(Path(cls.tmp.name) / "indexes")
        cls.embedder, cls.reranker = fake_embedder(), fake_reranker()
        build_bm25(cls.cfg, show_progress=False)
        build_dense(cls.cfg, cls.embedder, show_progress=False)
        cls.res = run_eval(cls.cfg, embedder=cls.embedder, reranker=cls.reranker)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_all_modes_have_sane_metrics(self):
        self.assertEqual(list(self.res.table), list(MODES))
        self.assertEqual(len(self.res.qids), 18)
        for metrics in self.res.table.values():
            for name, v in metrics.items():
                self.assertGreaterEqual(v, 0.0, name)
                if name != "ms/q":
                    self.assertLessEqual(v, 1.0, name)
        self.assertIn("Hybrid + Rerank", format_table(self.res))

    def test_rerank_keeps_tail(self):
        fused = self.res.lists["hybrid_rrf"]
        reranked = self.res.lists["hybrid_rerank"]
        top_n = self.cfg["reranker"]["top_n"]
        for f, r in zip(fused, reranked):
            self.assertEqual(sorted(d for d, _ in f[:top_n]), sorted(d for d, _ in r[:top_n]))
            self.assertEqual(f[top_n:], r[top_n:])

    def test_failure_analysis(self):
        summary, rows = analyze_failures(self.res, self.cfg, k=1)
        cats = sum(summary.get(c, 0) for c in ("召回失败", "截断失败", "排序失败"))
        self.assertEqual(cats, summary["failed"])
        self.assertEqual(len(rows), summary["failed"])
        for row in rows:
            self.assertIn(row["category"], ("召回失败", "截断失败", "排序失败"))

    def test_latency_and_save(self):
        latency = measure_latency(self.cfg, self.res, n=5, warmup=1)
        for stage in ("bm25", "dense", "fusion", "rerank", "total"):
            self.assertIn(stage, latency)
            self.assertLessEqual(latency[stage]["p50"], latency[stage]["p95"] + 1e-9)
        out = Path(self.tmp.name) / "results"
        md = save_results(self.res, self.cfg, out, tag="demo", failures=analyze_failures(self.res, self.cfg), latency=latency)
        self.assertTrue(md.exists())
        self.assertTrue(md.with_suffix(".json").exists())
        data = json.loads(md.with_suffix(".json").read_text(encoding="utf-8"))
        self.assertEqual(data["n_queries"], 18)

    def test_pipeline_ask(self):
        pipe = RAGPipeline(self.cfg, embedder=self.embedder, reranker=self.reranker, generator=Generator({}))
        res = pipe.ask("宿舍里可以用电热毯吗")
        self.assertIsNone(res["answer"])
        self.assertEqual(len(res["contexts"]), self.cfg["generation"]["context_top_k"])
        self.assertEqual(res["final"][0][0], "d14")

        class EchoGenerator:
            available = True

            def answer(self, question, contexts):
                return f"{question}|{len(contexts)}"

        pipe.generator = EchoGenerator()
        self.assertEqual(pipe.ask("校园卡丢了怎么办")["answer"], "校园卡丢了怎么办|3")

    def test_rerank_head_direct(self):
        docs = {"a": {"id": "a", "text": "苹果"}, "b": {"id": "b", "text": "香蕉苹果"}, "c": {"id": "c", "text": "梨"}}
        out = rerank_head(self.reranker, ["香蕉"], [[("a", 3.0), ("b", 2.0), ("c", 1.0)]], docs, top_n=2)[0]
        self.assertEqual([d for d, _ in out], ["b", "a", "c"])

    def test_bm25_only_eval_without_dense(self):
        res = run_eval(self.cfg, modes=["bm25"])
        self.assertEqual(list(res.table), ["bm25"])
        self.assertIsNone(res.dense)
        self.assertEqual(analyze_failures(res, self.cfg, k=1)[0]["mode"], "bm25")
        self.assertEqual(set(measure_latency(self.cfg, res, n=3, warmup=1)), {"bm25", "total"})

    def test_dense_only_latency_without_bm25_index(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = demo_config(Path(d) / "indexes")
            build_dense(cfg, self.embedder, show_progress=False)  # 故意不建 BM25
            res = run_eval(cfg, embedder=self.embedder, reranker=self.reranker, modes=["dense", "dense_rerank"])
            latency = measure_latency(cfg, res, n=3, warmup=1)
            self.assertEqual(set(latency), {"dense", "rerank", "total"})

    def test_results_files_do_not_collide(self):
        out = Path(self.tmp.name) / "results_twice"
        a = save_results(self.res, self.cfg, out, tag="same")
        b = save_results(self.res, self.cfg, out, tag="same")
        self.assertNotEqual(a, b)


class TestPrepAndGenerator(unittest.TestCase):
    def test_prepare_docs_handles_gbk(self):
        with tempfile.TemporaryDirectory() as d:
            docs = Path(d) / "docs"
            (docs / "sub").mkdir(parents=True)
            (docs / "a.txt").write_text("第一段。" * 200, encoding="utf-8")
            (docs / "sub" / "b.md").write_bytes("第二个文件，GBK 编码。".encode("gbk"))
            stats = prepare_docs(docs, Path(d) / "out", chunk_size=100, overlap=20)
            rows = [json.loads(x) for x in (Path(d) / "out" / "corpus.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(stats["files"], 2)
            self.assertTrue(any(r["id"].startswith("sub/b.md#") and "GBK" in r["text"] for r in rows))
            self.assertTrue(all(len(r["text"]) <= 100 for r in rows))

    def test_make_splits_for_own_data(self):
        with tempfile.TemporaryDirectory() as d:
            write_jsonl(Path(d) / "queries.jsonl", [{"id": f"q{i}", "text": f"问题{i}"} for i in range(10)])
            write_qrels(Path(d) / "qrels.tsv", {f"q{i}": {f"d{i}": 1} for i in range(10)})
            stats = make_splits_for(d, n_test=4, n_dev=2)
            self.assertEqual(stats, {"test": 4, "dev": 2, "train": 4})
            self.assertEqual(len(load_split_ids(d, "test")), 4)

    def test_generator_messages_and_availability(self):
        msgs = build_messages("问题？", ["资料一", "资料二" * 400], max_chars_per_doc=10)
        self.assertEqual(msgs[0]["role"], "system")
        self.assertIn("[2] 资料二资料二资料二资\n", msgs[1]["content"])  # 截断到 10 个字
        self.assertNotIn("资料二" * 4, msgs[1]["content"])
        os.environ.pop("RAG_TEST_KEY", None)
        self.assertFalse(Generator({"base_url": "https://x", "model": "m", "api_key_env": "RAG_TEST_KEY"}).available)
        self.assertTrue(Generator({"base_url": "http://localhost:11434/v1", "model": "m", "api_key_env": "RAG_TEST_KEY"}).available)


class TestCLI(unittest.TestCase):
    """命令行脚本的 BM25 路径不依赖 torch，可以直接跑。"""

    def test_build_and_evaluate_bm25(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = demo_config(Path(d) / "indexes")
            cfg_path = Path(d) / "cfg.yaml"
            cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
            env = {**os.environ, "PYTHONIOENCODING": "utf-8"}  # Windows 管道默认 GBK，统一成 utf-8
            run = lambda *a: subprocess.run([sys.executable, *a], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", env=env)  # noqa: E731
            r1 = run("scripts/build_index.py", "--config", str(cfg_path), "--only", "bm25")
            self.assertEqual(r1.returncode, 0, r1.stderr)
            r2 = run("scripts/evaluate.py", "--config", str(cfg_path), "--modes", "bm25", "--out", str(Path(d) / "res"))
            self.assertEqual(r2.returncode, 0, r2.stderr)
            self.assertIn("BM25 (jieba)", r2.stdout)
            self.assertTrue(list((Path(d) / "res").glob("*.md")))
            r3 = run("scripts/finetune_embedder.py")
            self.assertNotEqual(r3.returncode, 0)


if __name__ == "__main__":
    unittest.main()
