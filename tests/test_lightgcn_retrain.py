"""Small CPU fixtures for train+validation retraining; run with unittest."""

from contextlib import redirect_stdout, redirect_stderr
import io
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import polars as pl
import torch

from lightgcn.data import load_dataset, normalized_adjacency, sample_negatives
from lightgcn.model import LightGCN
from lightgcn import train, evaluate_checkpoint, retrain


class RetrainingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.data_dir = self.directory / "data"
        self.data_dir.mkdir()
        self.write_split("train", [(item % 3, item) for item in range(36)])
        self.write_split("valid", [(0, 1), (1, 2), (2, 0)])
        self.write_split("test", [(0, 2), (1, 0), (2, 1)])
        (self.data_dir / "manifest.json").write_text(json.dumps({
            "final_split_counts": {"train": 36, "valid": 3, "test": 3}}))

    def write_split(self, name: str, pairs: list[tuple[int, int]]) -> None:
        pl.DataFrame({"userId": [u + 10 for u, _ in pairs],
                      "movieId": [i + 100 for _, i in pairs]}).write_parquet(self.data_dir / f"{name}.parquet")

    def model(self, data) -> LightGCN:
        return LightGCN(data.n_users, data.n_items, 4, 2, (0.975, 0.0125, 0.0125))

    def arguments(self, output: str, *extra: str) -> list[str]:
        return ["train", "--data-dir", str(self.data_dir), "--output-dir", str(self.directory / output),
                "--layers", "2", "--embedding-dim", "4", "--layer-weights", "0.975,0.0125,0.0125",
                "--device", "cpu", "--batch-size", "20", *extra]

    def test_combined_graph_mapping_offsets_and_negatives(self) -> None:
        original = load_dataset(self.data_dir)
        data = load_dataset(self.data_dir, "train-valid")
        np.testing.assert_array_equal(data.user_ids, original.user_ids)
        np.testing.assert_array_equal(data.item_ids, original.item_ids)
        self.assertEqual(len(data.train.users), 39)
        self.assertTrue(np.all(np.diff(data.train_keys) > 0))
        np.testing.assert_array_equal(np.diff(data.train_offsets), [13, 13, 13])
        graph = normalized_adjacency(data, torch.device("cpu")).to_dense().numpy()
        self.assertEqual(np.count_nonzero(graph), 78)
        self.assertAlmostEqual(graph[0, data.n_users + 1], 1 / math.sqrt(13 * 2), places=6)
        self.assertEqual(graph[0, data.n_users + 2], 0)  # Test-only edge.
        np.testing.assert_array_equal(graph, graph.T)
        users = np.tile(np.arange(3, dtype=np.int32), 1000)
        negatives = sample_negatives(users, data.train_bits, data.n_items, np.random.default_rng(42))
        self.assertFalse(np.isin(users.astype(np.int64) * data.n_items + negatives, data.train_keys).any())

    def test_rejects_overlap_duplicates_unknown_ids_and_bad_manifest(self) -> None:
        for pairs, error in [([(0, 0), (1, 2), (2, 0)], "Overlapping"),
                             ([(0, 1), (0, 1), (2, 0)], "Duplicate"),
                             ([(0, 99), (1, 2), (2, 0)], "Unknown item")]:
            with self.subTest(error=error):
                self.write_split("valid", pairs)
                with self.assertRaisesRegex(ValueError, error):
                    load_dataset(self.data_dir, "train-valid")
        self.write_split("valid", [(0, 1), (1, 2), (2, 0)])
        (self.data_dir / "manifest.json").write_text(json.dumps({"final_split_counts": {"train": 99}}))
        with self.assertRaisesRegex(ValueError, "count does not match manifest"):
            load_dataset(self.data_dir)

    def test_dense_users_cannot_sample_negatives(self) -> None:
        self.write_split("train", [(0, 0), (1, 1)])
        self.write_split("valid", [(0, 1)])
        self.write_split("test", [])
        (self.data_dir / "manifest.json").unlink()
        with self.assertRaisesRegex(ValueError, "no unobserved item"):
            load_dataset(self.data_dir, "train-valid")

    def test_warmup_preserves_adam_moments_and_updates_weights(self) -> None:
        data = load_dataset(self.data_dir, "train-valid")
        model = self.model(data)
        graph = normalized_adjacency(data, torch.device("cpu"))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        rng = np.random.default_rng(42)
        before = model.user_embedding.weight.detach().clone()
        records = [train.train_epoch(model, graph, data, optimizer, rng, 20, 0.001, 0.001,
                                     step_offset=2 * epoch, warmup_steps=6) for epoch in range(3)]
        self.assertAlmostEqual(records[0]["learning_rate_start"], 0.001 / 6)
        self.assertAlmostEqual(records[-1]["learning_rate_end"], 0.001)
        self.assertFalse(torch.equal(before, model.user_embedding.weight))
        moments = optimizer.state[model.user_embedding.weight]["exp_avg"]
        self.assertTrue(torch.count_nonzero(moments) > 0)
        train.train_epoch(model, graph, data, optimizer, rng, 20, 0.001, 0.001,
                          step_offset=6, warmup_steps=6)
        self.assertIs(moments, optimizer.state[model.user_embedding.weight]["exp_avg"])
        self.assertEqual(optimizer.state[model.user_embedding.weight]["step"].item(), 8)
        self.assertEqual(train.warmup_learning_rate(0.001, 7, 6), 0.001)
        self.assertEqual(train.warmup_learning_rate(0.001, 1, 0), 0.001)

    def test_stopping_small_and_accumulated_improvements(self) -> None:
        stopping = train.TrainLossStopping(3, 0.0001)
        self.assertEqual(stopping.update(1.0), (True, False))
        self.assertEqual(stopping.update(0.99995), (True, False))
        self.assertEqual(stopping.update(0.99989), (True, False))
        self.assertEqual(stopping.stale_epochs, 0)
        self.assertEqual(stopping.update(1.0), (False, False))
        self.assertEqual(stopping.update(1.0), (False, False))
        self.assertEqual(stopping.update(1.0), (False, True))
        for loss in (float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "Non-finite"):
                stopping.update(loss)

    def test_nonfinite_loss_aborts_before_optimizer_step(self) -> None:
        data = load_dataset(self.data_dir)
        model = self.model(data)
        with torch.no_grad():
            model.user_embedding.weight.fill_(float("nan"))
        optimizer = torch.optim.Adam(model.parameters())
        with self.assertRaisesRegex(ValueError, "Non-finite training loss"):
            train.train_epoch(model, normalized_adjacency(data, torch.device("cpu")), data,
                              optimizer, np.random.default_rng(42), 20, 0.001, 0.001)
        self.assertFalse(optimizer.state)

    def test_warmup_excluded_from_selection_and_best_checkpoint_restored(self) -> None:
        data = load_dataset(self.data_dir)
        source = self.directory / "source.pt"
        train.save_checkpoint(source, self.model(data), data,
                              {"layers": 2, "embedding_dim": 4, "layer_weights": [0.975, 0.0125, 0.0125]}, 6)
        losses = [0.01, 0.02, 0.03, 1.0, 0.99995, 0.99996, 0.99997]
        calls = []

        def fake_epoch(model, *args, **kwargs):
            epoch = len(calls) + 1
            calls.append(epoch)
            with torch.no_grad():
                model.user_embedding.weight.fill_(epoch)
            return {"train_loss": losses[epoch - 1], "optimizer_steps": epoch * 2,
                    "learning_rate_start": 0.001, "learning_rate_end": 0.001}

        def fake_evaluate(model, adjacency, loaded, target, offsets, mask_valid, **kwargs):
            self.assertIs(target, loaded.test)
            self.assertEqual(loaded.train_split, "train-valid")
            self.assertEqual(model.user_embedding.weight[0, 0].item(), 5)
            return {"users": 3, "recall@20": 1.0, "ndcg@20": 1.0}

        argv = self.arguments("early", "--train-split", "train-valid", "--init-checkpoint", str(source),
                              "--warmup-epochs", "3", "--max-epochs", "10",
                              "--early-stopping", "train-loss", "--patience", "3")
        with patch("sys.argv", argv), patch.object(train, "train_epoch", side_effect=fake_epoch), \
                patch.object(train, "evaluate", side_effect=fake_evaluate) as evaluation, redirect_stdout(io.StringIO()):
            train.main()
        self.assertEqual(evaluation.call_count, 1)
        metrics = json.loads((self.directory / "early/metrics.json").read_text())
        self.assertEqual(metrics["epochs_completed"], 7)
        self.assertEqual(metrics["selected_epoch"], 5)
        self.assertEqual(metrics["selected_main_epoch"], 2)
        self.assertEqual(metrics["stop_reason"], "early_stopping")
        history = json.loads((self.directory / "early/history.json").read_text())
        self.assertTrue(all(not row["checkpoint_saved"] for row in history[:3]))
        saved = torch.load(self.directory / "early/best.pt", weights_only=True)
        self.assertEqual(saved["config"]["source_epoch"], 6)
        self.assertIn("optimizer_state_dict", saved)
        self.assertIsNone(saved["validation"])

    def test_epoch_budget_fixed_training_and_real_checkpoint_roundtrip(self) -> None:
        actual_train_epoch = train.train_epoch

        def decreasing_loss(*args, **kwargs):
            record = actual_train_epoch(*args, **kwargs)
            record["train_loss"] = 1 / record["optimizer_steps"]
            return record

        for name, extra, expected in [
            ("scratch", ["--max-epochs", "6", "--early-stopping", "none"], 6),
            ("budget", ["--max-epochs", "10", "--warmup-epochs", "3", "--early-stopping", "train-loss",
                        "--patience", "3", "--min-delta", "0"], 13),
        ]:
            argv = self.arguments(name, "--train-split", "train-valid", "--skip-test", *extra)
            # Exercise real Adam updates, with controlled losses for the budget assertion.
            epoch_function = decreasing_loss if name == "budget" else actual_train_epoch
            with patch("sys.argv", argv), patch.object(train, "train_epoch", side_effect=epoch_function), \
                    redirect_stdout(io.StringIO()):
                train.main()
            metrics = json.loads((self.directory / name / "metrics.json").read_text())
            self.assertEqual(metrics["epochs_completed"], expected)
            self.assertEqual(metrics["selected_epoch"], expected)
            self.assertEqual(metrics["stop_reason"], "max_epochs")
            saved = torch.load(self.directory / name / "best.pt", weights_only=True)
            self.assertTrue(saved["optimizer_state_dict"]["state"])
        output = self.directory / "evaluation.json"
        argv = ["evaluate", "--checkpoint", str(self.directory / "scratch/best.pt"),
                "--data-dir", str(self.data_dir), "--output", str(output), "--device", "cpu"]
        with patch("sys.argv", argv), redirect_stdout(io.StringIO()):
            evaluate_checkpoint.main()
        result = json.loads(output.read_text())
        self.assertNotIn("full_validation", result)
        self.assertEqual(result["test"]["users"], 3)
        self.assertEqual(result["train_split"], "train-valid")

    def test_legacy_checkpoint_and_mismatch_checks(self) -> None:
        data = load_dataset(self.data_dir)
        model = self.model(data)
        source = self.directory / "legacy.pt"
        train.save_checkpoint(source, model, data,
                              {"layers": 2, "embedding_dim": 4, "layer_weights": list(model.layer_weights)},
                              6, {"recall@20": 0.1})
        saved = torch.load(source, weights_only=True)
        train.validate_checkpoint(saved, data, model)
        with self.assertRaisesRegex(ValueError, "architecture"):
            train.validate_checkpoint(saved, data, LightGCN(3, 36, 8, 2))
        changed = {**saved, "user_ids": saved["user_ids"] + 1}
        with self.assertRaisesRegex(ValueError, "mapping"):
            train.validate_checkpoint(changed, data, model)
        argv = ["evaluate", "--checkpoint", str(source), "--data-dir", str(self.data_dir),
                "--output", str(self.directory / "legacy_metrics.json"), "--device", "cpu"]
        with patch("sys.argv", argv), redirect_stdout(io.StringIO()):
            evaluate_checkpoint.main()
        result = json.loads((self.directory / "legacy_metrics.json").read_text())
        self.assertEqual(result["full_validation"]["users"], 3)
        self.assertEqual(result["test"]["users"], 3)

    def test_masking_and_ranking_metrics(self) -> None:
        data = load_dataset(self.data_dir, "train-valid")
        model = self.model(data)
        # One dimension per user provides distinct, deterministic rankings.
        user_vectors = torch.eye(3)
        item_vectors = torch.zeros(36, 3)
        for user in range(3):
            item_vectors[data.train.items[data.train.users == user], user] = 100
            item_vectors[data.test.items[data.test.users == user], user] = 10
        with patch.object(model, "propagate", return_value=(user_vectors, item_vectors)):
            result = train.evaluate(model, torch.empty(0), data, data.test, data.test_offsets, True)
            self.assertEqual(result, {"users": 3, "recall@20": 1.0, "ndcg@20": 1.0})
            with self.assertRaisesRegex(ValueError, "already been used"):
                train.evaluate(model, torch.empty(0), data, data.valid, data.valid_offsets, False)
        original = load_dataset(self.data_dir)
        with patch.object(model, "propagate", return_value=(user_vectors, item_vectors)):
            self.assertEqual(train.evaluate(model, torch.empty(0), original, original.test,
                                            original.test_offsets, True), result)
            unmasked = train.evaluate(model, torch.empty(0), original, original.test, original.test_offsets, False)
            self.assertLess(unmasked["ndcg@20"], 1.0)

    def test_default_validation_and_invalid_options(self) -> None:
        with patch("sys.argv", self.arguments("legacy_run", "--max-epochs", "1", "--skip-test")), \
                redirect_stdout(io.StringIO()):
            train.main()
        metrics = json.loads((self.directory / "legacy_run/metrics.json").read_text())
        self.assertIn("full_validation", metrics)
        self.assertEqual(metrics["selection_criterion"], "validation")
        for options in [("--train-split", "train-valid"), ("--warmup-epochs", "-1"), ("--min-delta", "nan")]:
            with patch("sys.argv", self.arguments("invalid", *options)), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    train.main()
                self.assertEqual(error.exception.code, 2)

    def test_recall_and_ndcg_denominators_with_multiple_targets(self) -> None:
        self.write_split("test", [(0, 2), (0, 5), (1, 0), (2, 1)])
        (self.data_dir / "manifest.json").unlink()
        data = load_dataset(self.data_dir, "train-valid")
        model = self.model(data)
        user_vectors = torch.eye(3)
        item_vectors = torch.zeros(36, 3)
        for user in range(3):
            item_vectors[data.train.items[data.train.users == user], user] = 100
        item_vectors[2, 0] = item_vectors[0, 1] = item_vectors[1, 2] = 10
        item_vectors[5, 0] = -1  # Outside top 20: only one of user 0's two targets hits.
        with patch.object(model, "propagate", return_value=(user_vectors, item_vectors)):
            result = train.evaluate(model, torch.empty(0), data, data.test, data.test_offsets, True)
        self.assertAlmostEqual(result["recall@20"], (0.5 + 1 + 1) / 3)
        self.assertAlmostEqual(result["ndcg@20"], (1 / (1 + 1 / math.log2(3)) + 2) / 3)

    def test_canonical_audit_hash_matches_preparation_format(self) -> None:
        path = self.directory / "hash_fixture.parquet"
        pl.DataFrame({"userId": [11, 10], "movieId": [102, 101], "rating": [4.5, 5.0],
                      "timestamp": [300, 200]}).write_parquet(path)
        expected = hashlib.sha256(b"10,101,5.0,200\n11,102,4.5,300\n").hexdigest()
        self.assertEqual(retrain.canonical_hash(path), expected)

    def test_report_uses_measured_metrics_and_correct_deltas(self) -> None:
        baseline = {"selected_epoch": 6, "test": {"users": 3, "recall@20": 0.1, "ndcg@20": 0.2}}
        for name, values in [("scratch", (0.11, 0.18)), ("finetune", (0.12, 0.21))]:
            directory = self.directory / name
            directory.mkdir()
            metrics = {"test": {"users": 3, "recall@20": values[0], "ndcg@20": values[1]},
                       "warmup_epochs_completed": 3 if name == "finetune" else 0,
                       "main_epochs_completed": 6, "selected_main_epoch": 6,
                       "training_seconds": 1.5, "test_seconds": 0.1, "total_seconds": 2.0,
                       "selected_train_loss": 0.5, "stop_reason": "max_epochs"}
            (directory / "metrics.json").write_text(json.dumps(metrics))
            (directory / "config.json").write_text(json.dumps({"device_used": "cpu"}))
        retrain.write_report(self.directory, baseline, {"scratch": ["python"], "finetune": ["python"]},
                             {"combined_training_interactions": 39, "users": 3, "items": 36, "eligible_test_users": 3})
        result = json.loads((self.directory / "summary.json").read_text())
        self.assertAlmostEqual(result["deltas_from_baseline"]["scratch"]["recall@20"]["absolute"], 0.01)
        self.assertAlmostEqual(result["deltas_from_baseline"]["scratch"]["ndcg@20"]["relative_percent"], -10)
        self.assertAlmostEqual(result["deltas_from_baseline"]["finetune"]["recall@20"]["relative_percent"], 20)
        self.assertIn("0.110000", (self.directory / "report.md").read_text())


if __name__ == "__main__":
    unittest.main()
