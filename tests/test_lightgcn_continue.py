"""CPU checks for resuming Adam, sampling RNG, and train-loss early stopping."""

from contextlib import redirect_stdout
import io
import json
import unittest
from unittest.mock import patch

import numpy as np
import torch

import test_lightgcn_retrain as fixtures
from lightgcn import continue_finetune, train
from lightgcn.data import load_dataset, normalized_adjacency


class ContinuationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = fixtures.RetrainingTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.directory = self.fixture.directory
        self.data = load_dataset(self.fixture.data_dir, "train-valid")
        self.graph = normalized_adjacency(self.data, torch.device("cpu"))
        self.model = self.fixture.model(self.data)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.rng = np.random.default_rng(42)
        self.record = {}
        for epoch in range(2):
            self.record = train.train_epoch(self.model, self.graph, self.data, self.optimizer, self.rng,
                                            20, 0.001, 0.001, step_offset=epoch * 2)
        self.record.update({"epoch": 2, "main_epoch": 2, "phase": "main", "stale_epochs": 0,
                            "train_loss": 0.5})
        self.config = {"layers": 2, "embedding_dim": 4, "layer_weights": list(self.model.layer_weights),
                       "learning_rate": 0.001, "l2": 0.001, "batch_size": 20, "seed": 42,
                       "train_split": "train-valid", "early_stopping": "train-loss", "patience": 3,
                       "min_delta": 0.0001, "warmup_epochs": 0, "n_items": 36, "n_users": 3}
        self.source = self.directory / "source.pt"
        train.save_checkpoint(self.source, self.model, self.data, self.config, 2,
                              optimizer=self.optimizer, training=self.record)

    def argv(self, output: str, *extra: str) -> list[str]:
        return ["continue", "--checkpoint", str(self.source), "--data-dir", str(self.fixture.data_dir),
                "--output-dir", str(self.directory / output), "--device", "cpu", *extra]

    def test_saved_and_replayed_rng_match_uninterrupted_training(self) -> None:
        checkpoint = torch.load(self.source, weights_only=True)
        rng, method = continue_finetune.restore_sampling_rng(checkpoint, self.data)
        self.assertEqual(method, "replayed_sampling_only")
        self.assertEqual(rng.bit_generator.state, self.rng.bit_generator.state)
        train.save_checkpoint(self.source, self.model, self.data, self.config, 2,
                              optimizer=self.optimizer, training=self.record, rng=self.rng)
        checkpoint = torch.load(self.source, weights_only=True)
        direct, method = continue_finetune.restore_sampling_rng(checkpoint, self.data)
        self.assertEqual(method, "saved_state")
        self.assertEqual(direct.bit_generator.state, self.rng.bit_generator.state)
        resumed_model = self.fixture.model(self.data)
        resumed_model.load_state_dict(checkpoint["state_dict"])
        resumed_optimizer = torch.optim.Adam(resumed_model.parameters())
        resumed_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        for old, restored in zip(self.optimizer.state.values(), resumed_optimizer.state.values()):
            for key in ("step", "exp_avg", "exp_avg_sq"):
                torch.testing.assert_close(old[key], restored[key], rtol=0, atol=0)
        uninterrupted = train.train_epoch(self.model, self.graph, self.data, self.optimizer, self.rng,
                                          20, 0.001, 0.001, step_offset=4)
        resumed = train.train_epoch(resumed_model, self.graph, self.data, resumed_optimizer, rng,
                                    20, 0.001, 0.001, step_offset=4)
        self.assertEqual(uninterrupted, resumed)
        for key, value in self.model.state_dict().items():
            torch.testing.assert_close(value, resumed_model.state_dict()[key], rtol=0, atol=0)
        self.assertEqual({int(state["step"].item()) for state in resumed_optimizer.state.values()}, {6})

    def test_no_improvement_keeps_source_checkpoint_and_stops_after_three(self) -> None:
        def fake_epoch(model, *args, **kwargs):
            with torch.no_grad():
                model.user_embedding.weight.fill_(99)
            return {"train_loss": 0.6, "optimizer_steps": kwargs["step_offset"] + 2}

        def fake_evaluate(model, graph, data, target, offsets, mask_valid):
            self.assertIs(target, data.test)
            torch.testing.assert_close(model.user_embedding.weight, self.model.user_embedding.weight)
            return {"users": 3, "recall@20": 0.2, "ndcg@20": 0.3}

        with patch("sys.argv", self.argv("early")), patch.object(continue_finetune, "train_epoch", side_effect=fake_epoch), \
                patch.object(continue_finetune, "evaluate", side_effect=fake_evaluate) as evaluation, \
                redirect_stdout(io.StringIO()):
            continue_finetune.main()
        metrics = json.loads((self.directory / "early/metrics.json").read_text())
        self.assertEqual(metrics["additional_epochs_completed"], 3)
        self.assertEqual(metrics["selected_epoch"], 2)
        self.assertEqual(metrics["selected_additional_epoch"], 0)
        self.assertEqual(metrics["stop_reason"], "early_stopping")
        self.assertEqual(evaluation.call_count, 1)
        selected = torch.load(self.directory / "early/best.pt", weights_only=True)
        self.assertIn("numpy_rng_state", selected)
        self.assertEqual(selected["config"]["optimizer_initialization"], "restored_adam")

    def test_budget_continues_steps_and_cumulative_epoch_numbers(self) -> None:
        actual_epoch = train.train_epoch

        def decreasing_epoch(*args, **kwargs):
            record = actual_epoch(*args, **kwargs)
            record["train_loss"] = 1 / record["optimizer_steps"]
            return record

        with patch("sys.argv", self.argv("budget")), \
                patch.object(continue_finetune, "train_epoch", side_effect=decreasing_epoch), redirect_stdout(io.StringIO()):
            continue_finetune.main()
        metrics = json.loads((self.directory / "budget/metrics.json").read_text())
        self.assertEqual(metrics["additional_epochs_completed"], 10)
        self.assertEqual(metrics["selected_epoch"], 12)
        self.assertEqual(metrics["selected_main_epoch"], 12)
        self.assertEqual(metrics["warmup_epochs_completed"], 0)
        selected = torch.load(self.directory / "budget/best.pt", weights_only=True)
        self.assertEqual({int(state["step"].item()) for state in selected["optimizer_state_dict"]["state"].values()}, {24})
        self.assertEqual(selected["training"]["continued_epoch"], 10)
        self.assertIn("early_stopping_state", selected["training"])
        restored = continue_finetune.restore_stopping(selected, self.directory / "budget/best.pt", 3, 0.0001)
        self.assertEqual(restored.best_loss, metrics["selected_train_loss"])
        rng, method = continue_finetune.restore_sampling_rng(selected, self.data)
        self.assertEqual(method, "saved_state")
        self.assertEqual(rng.bit_generator.state, selected["numpy_rng_state"])

    def test_history_restores_significant_improvement_reference(self) -> None:
        source = torch.load(self.source, weights_only=True)
        source["training"].update({"train_loss": 0.49995, "stale_epochs": 1})
        (self.directory / "history.json").write_text(json.dumps([
            {"epoch": 1, "phase": "main", "train_loss": 0.5},
            {"epoch": 2, "phase": "main", "train_loss": 0.49995},
            {"epoch": 3, "phase": "main", "train_loss": 0.4},
        ]))
        stopping = continue_finetune.restore_stopping(source, self.source, 3, 0.0001)
        self.assertEqual(stopping.best_loss, 0.49995)
        self.assertEqual(stopping.improvement_loss, 0.5)
        self.assertEqual(stopping.stale_epochs, 1)
        self.assertEqual(stopping.update(0.49989), (True, False))
        self.assertEqual(stopping.stale_epochs, 0)

    def test_missing_adam_is_rejected(self) -> None:
        source = torch.load(self.source, weights_only=True)
        del source["optimizer_state_dict"]
        torch.save(source, self.source)
        with patch("sys.argv", self.argv("missing")), redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, "without saved Adam"):
                continue_finetune.main()


if __name__ == "__main__":
    unittest.main()
