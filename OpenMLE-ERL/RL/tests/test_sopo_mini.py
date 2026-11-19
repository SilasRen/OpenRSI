import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sopo_mini"))
sys.path.insert(0, str(ROOT))

try:
    import sklearn  # noqa: F401
    import pandas  # noqa: F401
    HAVE_DEPS = True
except ImportError:
    HAVE_DEPS = False


@unittest.skipUnless(HAVE_DEPS, "needs pandas and scikit-learn")
class SopoMiniTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import tasks
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name) / "tasks"
        tasks.main(["--out", str(cls.root), "--train-per-kind", "1", "--eval-per-kind", "1"])

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_labels_never_in_public_files(self):
        from tasks import load_tasks
        for t in load_tasks(self.root, "eval"):
            test_cols = (t.public_dir / "test.csv").read_text().splitlines()[0].split(",")
            self.assertNotIn("target", test_cols)
            self.assertTrue(t.private()["y_test"])

    def test_executor_scores_ladder_and_rejects_forbidden(self):
        from executor import run_program
        from policy import _FAKE
        from tasks import load_tasks
        t = load_tasks(self.root, "train")[0]
        crash, const, gbm = (run_program(_FAKE[k], t, 60) for k in ("crash", "constant", "gbm"))
        self.assertFalse(crash.eligible)
        self.assertEqual(crash.category, "valid")
        self.assertAlmostEqual(const.utility, 0.0, places=6)
        self.assertGreater(gbm.utility, 0.9)
        hack = run_program("import subprocess\n", t, 10)
        self.assertEqual(hack.category, "hack")
        peek = run_program("open('../_private/x.json')\n", t, 10)
        self.assertEqual(peek.category, "hack")

    def test_end_to_end_fake_pipeline(self):
        import evaluate
        import report
        import train
        out = Path(self.tmp.name)
        for name, extra in [("control", ["--sopo-steps", "0"]), ("sopo", ["--sopo-steps", "2", "--curriculum"])]:
            train.main(["--tasks", str(self.root), "--out", str(out / name), "--fake-policy", "--steps", "2",
                        "--groups", "2", "--group-size", "4", "--workers", "4", "--timeout", "60", *extra])
            evaluate.main(["--tasks", str(self.root), "--model", str(out / name / "final"), "--out",
                           str(out / "eval" / f"{name}.jsonl"), "--budget", "5", "--drafts", "2", "--seeds", "1",
                           "--workers", "4", "--timeout", "60"])
        logs = [json.loads(line) for line in (out / "sopo" / "train_log.jsonl").read_text().splitlines()]
        # SOPO executes 1 root + 2 lookahead programs per valid sibling, control only the root.
        ctrl = [json.loads(line) for line in (out / "control" / "train_log.jsonl").read_text().splitlines()]
        self.assertGreater(logs[-1]["executions"], ctrl[-1]["executions"])
        for g in logs[0]["groups"]:
            for r, b, u in zip(g["rewards"], g["branch_best"], g["root_utility"]):
                if u > 0:  # valid scored root: best-mode reward = best utility on its branch
                    self.assertAlmostEqual(r, b)
        report.main([str(out / "eval" / "control.jsonl"), str(out / "eval" / "sopo.jsonl"),
                     "--baseline", "control", "--out", str(out / "report"), "--budgets", "1,5"])
        summary = json.loads((out / "report" / "summary.json").read_text())
        self.assertIn("sopo", summary["vs_baseline"])


    def test_infer_credit_stats_and_fake_run(self):
        import infer_credit as ic
        self.assertAlmostEqual(ic.spearman([1, 2, 3], [3, 4, 9]), 1.0)
        self.assertIsNone(ic.spearman([1, 1, 1], [1, 2, 3]))
        self.assertEqual(ic.pair_accuracy([1.0, 0.0], [0.9, 0.1], 0.05), (1.0, 1))
        sib = {"r0": 0.5, "category": "valid", "utility": 0.5, "eligible": True,
               "discovery": [{"static_value": 0.5, "eligible": True}, {"static_value": 0.4, "eligible": True},
                             {"static_value": 0.9, "eligible": True}], "repeats": []}
        sig = ic.signals(sib, None, 2, [0.5])
        self.assertAlmostEqual(sig["best_K2"], 0.9)
        self.assertAlmostEqual(sig["best_K1"], 0.5)
        self.assertAlmostEqual(sig["credit_K2_l0.5"], 0.5 + 0.25 * 0.4)
        with tempfile.TemporaryDirectory() as d:
            ic.main(["--fake-policy", "--tasks", str(self.root), "--out", d, "--num-states", "2",
                     "--group-size", "3", "--repeats", "1", "--seed-drafts", "1", "--workers", "2"])
            summary = json.loads((Path(d) / "summary.json").read_text())
            self.assertIn("best_K2", summary["signals"])
            self.assertEqual(summary["groups_total"], 2)


if __name__ == "__main__":
    unittest.main()
