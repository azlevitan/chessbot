"""Profile isolation preserves exact inputs and reports beyond an early failure."""

import json
from pathlib import Path
import shutil

import numpy as np
import pytest
import torch

import export_probe_cases as probe


class ZeroModel(torch.nn.Module):
    def forward(self, planes):
        return torch.zeros(len(planes), 4168), torch.zeros(len(planes), 3)


def test_early_gate_failure_keeps_both_heads_same_row_and_all_real_cases(tmp_path, monkeypatch):
    run, evidence = tmp_path / "run", tmp_path / "evidence"
    run.mkdir()
    checkpoint = evidence / "backup/model.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"unchanged checkpoint fixture")
    model = Path(__file__).parents[2] / "engine/tests/fixtures/match_uniform.onnx"
    monkeypatch.setattr(probe, "load_checkpoint", lambda _: ZeroModel())
    monkeypatch.setattr(probe, "_export", lambda _, path, fold: shutil.copyfile(model, path))
    planes = torch.zeros(3, 18, 8, 8)
    mask = torch.zeros(3, 4168, dtype=torch.bool)
    mask[:, :2] = True
    monkeypatch.setattr(probe.exporter, "load_parity_positions",
                        lambda _: {"planes": planes, "legal_mask": mask})
    original, count = probe.compare_case, 0

    def failure_once(*args, **kwargs):
        nonlocal count
        count += 1
        result = original(*args, **kwargs)
        if count == 2:
            result.update(gate_pass=False, error="injected early random-7 failure")
        return result

    monkeypatch.setattr(probe, "compare_case", failure_once)
    report = probe.run_profile(checkpoint, evidence, run, "baseline", failing_row=2)
    assert report["completed"] is True and report["gate_pass"] is False
    rows = report["cases"]
    assert [row["name"] for row in rows[:5]] == [
        "random_1", "random_7", "same_row_2_cross", "same_row_2_torch", "same_row_2_ort"]
    assert len(rows) == 10
    assert all(row["completed"] for row in rows)
    assert set(rows[1]["heads"]) == {"policy_logits", "wdl_logits"}
    assert rows[1]["error"] == "injected early random-7 failure"
    real = [row for row in rows if row["source"] == "heldout"]
    assert [row["batch"] for row in real] == [1, 1, 1, 7, 32]
    assert all(row["rtol"] == 0 and row["gate_pass"] for row in real)
    directory = evidence / "profiles/baseline"
    generator = torch.Generator().manual_seed(0)
    expected_1 = torch.rand(1, 18, 8, 8, generator=generator).numpy()
    expected_7 = torch.rand(7, 18, 8, 8, generator=generator).numpy()
    np.testing.assert_array_equal(np.load(directory / "random_1.npy"), expected_1)
    np.testing.assert_array_equal(np.load(directory / "random_7.npy"), expected_7)
    assert len((directory / "cases.jsonl").read_text().splitlines()) == 10
    assert checkpoint.read_bytes() == b"unchanged checkpoint fixture"
    assert list(run.iterdir()) == []
    json.dumps(report, allow_nan=False)


def test_supported_export_toggle_is_isolated_to_experimental_artifact(tmp_path, monkeypatch):
    arguments, contracts = [], []
    monkeypatch.setattr(torch.onnx, "export", lambda *args, **kwargs: arguments.append(kwargs))
    monkeypatch.setattr(probe.exporter, "export_onnx", lambda model, path: torch.onnx.export(model, path))
    monkeypatch.setattr(probe.exporter, "check_contract", lambda path: contracts.append(path))
    normal, experimental = tmp_path / "normal.onnx", tmp_path / "unfolded.onnx"
    probe._export(None, normal, True)
    probe._export(None, experimental, False)
    assert arguments == [{}, {"do_constant_folding": False}]
    assert contracts == [normal, experimental]


@pytest.mark.parametrize("profile,threads,disabled", [
    ("baseline", None, False), ("single", 1, False), ("no_ort", 1, True),
    ("no_fold", 1, False), ("no_fold_no_ort", 1, True),
])
def test_runtime_controls_apply_only_to_the_chosen_profile(monkeypatch, profile, threads, disabled):
    seen = []
    monkeypatch.setattr(probe.ort, "InferenceSession", lambda path, **kwargs: seen.append(kwargs))
    probe._session(Path("unpublished.onnx"), profile)
    options = seen[0]["sess_options"]
    assert seen[0]["providers"] == ["CPUExecutionProvider"]
    if threads is None:
        assert options is None
    else:
        assert options.intra_op_num_threads == options.inter_op_num_threads == threads
        expected = probe.ort.GraphOptimizationLevel.ORT_DISABLE_ALL if disabled else probe.ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        assert options.graph_optimization_level == expected
