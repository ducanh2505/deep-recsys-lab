"""Device selection and RNG checkpoint recovery without downloading models."""

from contextlib import redirect_stdout
import io
import tempfile
from pathlib import Path
import types
import unittest
from unittest.mock import Mock, patch

import torch

from content_recsys import __main__ as cli
from content_recsys.common import device_for, memory_snapshot, restore_rng, rng_state, save_checkpoint


class DeviceTests(unittest.TestCase):
    def test_cuda_unavailable_fails_before_training(self) -> None:
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "CUDA must be available"):
                device_for("cuda")
            self.assertEqual(device_for("cpu"), torch.device("cpu"))

    def test_cli_passes_cuda_to_lora_training(self) -> None:
        trainer = types.ModuleType("content_recsys.train")
        trainer.run = Mock(return_value={"completed": True})
        argv = ["content_recsys", "train", "--device", "cuda", "--mode", "lora",
                "--layout", "single", "--loss", "bpr"]
        with patch("sys.argv", argv), patch.dict("sys.modules", {"content_recsys.train": trainer}), \
                patch("torch.cuda.is_available", return_value=True), \
                patch("torch.set_num_threads"), redirect_stdout(io.StringIO()):
            cli.main()
        config = trainer.run.call_args.args[0]
        self.assertEqual((config["device"], config["mode"]), ("cuda", "lora"))
        self.assertEqual(config["learning_rate"], 1e-4)

    def test_checkpoint_preserves_cuda_rng_on_cpu(self) -> None:
        device = torch.device("cuda:1")
        cuda_state = torch.tensor([1, 2, 3], dtype=torch.uint8)
        with patch("torch.cuda.get_rng_state", return_value=cuda_state) as capture:
            state = rng_state(device)
        capture.assert_called_once_with(device)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rng.pt"
            save_checkpoint(path, {"rng": state})
            loaded = torch.load(path, map_location="cpu", weights_only=True)
        with patch("torch.cuda.set_rng_state") as restore:
            restore_rng(loaded["rng"], device)
        torch.testing.assert_close(restore.call_args.args[0], cuda_state)
        self.assertEqual(restore.call_args.args[1], device)

    def test_memory_reports_cuda_usage_and_peak_in_mib(self) -> None:
        device = torch.device("cuda:1")
        with patch("torch.cuda.memory_allocated", return_value=2 * 2**20), \
                patch("torch.cuda.memory_reserved", return_value=4 * 2**20), \
                patch("torch.cuda.max_memory_allocated", return_value=3 * 2**20), \
                patch("torch.cuda.max_memory_reserved", return_value=5 * 2**20):
            memory = memory_snapshot(device)
        self.assertEqual(memory["cuda_allocated_mib"], 2.0)
        self.assertEqual(memory["cuda_reserved_mib"], 4.0)
        self.assertEqual(memory["cuda_peak_allocated_mib"], 3.0)
        self.assertEqual(memory["cuda_peak_reserved_mib"], 5.0)
        self.assertGreater(memory["process_peak_rss_mib"], 0)

    def test_cpu_rng_roundtrip_replays_random_values(self) -> None:
        device = torch.device("cpu")
        state = rng_state(device)
        expected = torch.rand(8)
        torch.rand(8)
        restore_rng(state, device)
        torch.testing.assert_close(torch.rand(8), expected, rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA GPU unavailable")
    def test_cuda_checkpoint_replays_cpu_and_gpu_random_values(self) -> None:
        device = device_for("cuda")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rng.pt"
            save_checkpoint(path, {"rng": rng_state(device)})
            expected_cpu = torch.rand(8)
            expected_cuda = torch.rand(8, device=device)
            torch.rand(8)
            torch.rand(8, device=device)
            saved = torch.load(path, map_location="cpu", weights_only=True)
            restore_rng(saved["rng"], device)
        torch.testing.assert_close(torch.rand(8), expected_cpu, rtol=0, atol=0)
        torch.testing.assert_close(torch.rand(8, device=device), expected_cuda, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
