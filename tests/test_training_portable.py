import csv
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from training.train_crnn import extract_train_characters


def test_preflight_resume_and_final_test(tmp_path):
    dataset = tmp_path / "dataset"
    fields = ["tier_id", "row_id", "parent_sample_id", "tier_name", "image_path", "label_path"]
    for split in ("train", "validation", "test"):
        path = dataset / split
        (path / "images").mkdir(parents=True)
        (path / "labels").mkdir()
        with (path / "metadata.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for i, tier in enumerate(("upper", "count", "lower")):
                cv2.imwrite(str(path / "images" / f"{i}.png"), np.full((64, 32), 255, np.uint8))
                (path / "labels" / f"{i}.txt").write_text("  A A" if i == 0 else "A", encoding="utf-8")
                writer.writerow(dict(tier_id=str(i), row_id="row", parent_sample_id=split,
                                     tier_name=tier, image_path=f"images/{i}.png", label_path=f"labels/{i}.txt"))
    assert extract_train_characters(dataset / "train/labels") == [" ", "A"]
    output = tmp_path / "run"
    prefix = [sys.executable, "-m", "training.train_crnn", "--dataset-root", str(dataset),
              "--output-root", str(output), "--batch-size", "3", "--num-workers", "0",
              "--checkpoint-mirror-dir", str(tmp_path / 'persistent')]
    # Keep tiny CPU integration practical without changing the production model.
    import os
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="")
    def run(extra):
        result = subprocess.run(prefix + extra, env=env, capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stdout + result.stderr
        return result.stdout
    # Memory-only probe must not create a misleading checkpoint.
    probe_log = run(["--epochs", "1", "--preflight-only", "--amp"])
    assert "AMP active: False" in probe_log
    assert "Memory probe:" in probe_log
    assert not (output / "last.pt").exists()
    quick_log = run(["--epochs", "1", "--preflight-only", "--amp", "--train-limit", "2", "--val-limit", "1"])
    assert 'Bounded preflight: sample vocabulary only' in quick_log
    assert not (output / 'last.pt').exists()
    run(["--epochs", "1", "--skip-test", "--max-batch-size", "1", "--stop-after-batches", "1"])
    partial = torch.load(output / "last.pt", weights_only=True)
    assert partial["epoch"] == 0
    assert partial["next_batch"] == partial["global_step"] == 1
    assert not (output / "best.pt").exists()
    # Fresh runtime: restore persistent state into a different active directory.
    from training.colab_checkpoints import restore
    restored = tmp_path / 'fresh_runtime'
    restore(tmp_path / 'persistent/checkpoints.zip', restored)
    prefix[prefix.index('--output-root') + 1] = str(restored)
    output = restored
    prefix += ["--max-batch-size", "1"]
    resumed_log = run(["--epochs", "1", "--skip-test", "--resume", str(output / "last.pt")])
    assert "Resume: epoch=0 next_batch=1 global_step=1" in resumed_log
    assert not (output / "test_results.json").exists()
    checkpoint = torch.load(output / "last.pt", weights_only=True)
    assert checkpoint["epoch"] == 1
    assert checkpoint["next_batch"] == 0
    assert checkpoint["global_step"] == 3
    assert checkpoint["char_to_idx"] == {" ": 1, "A": 2}
    assert checkpoint["idx_to_char"] == {0: "<BLANK>", 1: " ", 2: "A"}
    assert checkpoint["optimizer_state_dict"]["state"]
    assert set(checkpoint["history"][0]["validation"]["per_tier"]) == {"upper", "count", "lower"}
    run(["--epochs", "2", "--resume", str(output / "last.pt")])
    resumed = torch.load(output / "last.pt", weights_only=True)
    assert resumed["epoch"] == 2
    assert len(resumed["history"]) == 2
    for name in ("best.pt", "training_history.csv", "training_history.json", "test_results.json"):
        assert (output / name).exists()
    restore(tmp_path / 'persistent/checkpoints.zip', tmp_path / 'final_restore')
    restored_last = torch.load(tmp_path / 'final_restore/last.pt', weights_only=True)
    restored_best = torch.load(tmp_path / 'final_restore/best.pt', weights_only=True)
    assert restored_last['epoch'] == 2
    assert restored_last['best_val_cer'] == restored_best['best_val_cer']
    # A changed TRAIN mapping must fail before training resumes.
    (dataset / "train/labels/2.txt").write_text("Z")
    result = subprocess.run(prefix + ["--epochs", "3", "--resume", str(output / "last.pt")],
                            env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode != 0
    assert "vocabulary differs" in result.stderr
