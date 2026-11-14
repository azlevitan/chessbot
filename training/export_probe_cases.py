"""Isolated, read-only-checkpoint export profiles; no model publication or training."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import platform
from unittest.mock import patch

import numpy as np
import onnx
import onnxruntime as ort
import torch

import export_onnx as exporter
from checkpoint import load_checkpoint
from onnx_parity import PARITY_RTOL
from parity_diagnostics import compare_case

PROFILES = ("baseline", "single", "no_ort", "no_fold", "no_fold_no_ort")


def _hash_file(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _hash_array(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _safe_file(path, run):
    if path.resolve().is_relative_to(run):
        raise ValueError("diagnostic output must be outside the run directory")
    if path.is_symlink() or (path.exists() and (not path.is_file() or path.stat().st_nlink != 1)):
        raise ValueError(f"diagnostic output is not an independent regular file: {path}")
    return path


def _record(directory, run, cases, name, reference, observed, *, source, batch,
            mask=None, rtol=PARITY_RTOL, gate_case=True, input_hash=None):
    print(f"diagnostic {name}", flush=True)
    row = {"name": name, "source": source, "batch": batch, "gate_case": gate_case,
           "input_sha256": input_hash}
    pair = None
    try:
        pair = reference(), observed()
        row.update(compare_case(*pair, mask, rtol=rtol), completed=True)
    except Exception as error:
        row.update(completed=False, gate_pass=False, failure_kind="operational",
                   error=f"{type(error).__name__}: {error}", heads={})
    cases.append(row)
    path = _safe_file(directory / "cases.jsonl", run)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return pair


def _export(model, path, fold):
    if fold:
        exporter.export_onnx(model, path)
    else:
        original = torch.onnx.export
        def without_folding(*args, **kwargs):
            return original(*args, **dict(kwargs, do_constant_folding=False))
        with patch.object(torch.onnx, "export", without_folding):
            exporter.export_onnx(model, path)
    exporter.check_contract(path)


def _session(path, profile):
    options = None
    if profile != "baseline":
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_DISABLE_ALL if "no_ort" in profile
            else ort.GraphOptimizationLevel.ORT_ENABLE_ALL)
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


def run_profile(checkpoint: Path, evidence: Path, run: Path, profile: str, failing_row: int = 2) -> dict:
    """Run one profile in a fresh process, loading only the preserved checkpoint."""
    run, evidence, checkpoint = Path(run).resolve(), Path(evidence).resolve(), Path(checkpoint).resolve()
    if profile not in PROFILES or not 0 <= failing_row < 7:
        raise ValueError("invalid profile or failing row (expected 0..6)")
    if checkpoint != (evidence / "backup/model.pt").resolve():
        raise ValueError("worker requires evidence/backup/model.pt")
    directory = evidence / "profiles" / profile
    if directory.resolve().is_relative_to(run):
        raise ValueError("diagnostic evidence must be outside the run directory")
    directory.mkdir(parents=True, exist_ok=True)
    # A fresh process is required because torch inter-op threads can only be set once.
    if profile != "baseline":
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    settings, cases = {}, []
    result = {"format": "chessbot-export-profile-v1", "profile": profile,
              "completed": False, "gate_pass": False, "settings": settings, "cases": cases}
    try:
        # Match the normal exporter's input-loading order before model/export work.
        try:
            positions = exporter.load_parity_positions(run / "supervised_val")
            positions_error = None
        except Exception as error:
            positions, positions_error = None, error
        model = load_checkpoint(checkpoint).eval().cpu()
        settings.update(torch_version=torch.__version__, ort_version=ort.__version__,
                        onnx_version=onnx.__version__, numpy_version=np.__version__,
                        python=platform.python_version(), os=platform.system(), arch=platform.machine(),
                        torch_threads=torch.get_num_threads(), torch_interop_threads=torch.get_num_interop_threads(),
                        checkpoint_sha256=_hash_file(checkpoint), constant_folding_requested="no_fold" not in profile,
                        torch_mkldnn_available=torch.backends.mkldnn.is_available(),
                        torch_mkldnn_enabled=torch.backends.mkldnn.enabled,
                        torch_nnpack_available=torch.backends.nnpack.is_available(),
                        thread_environment={key: os.environ.get(key) for key in (
                            "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")})
        path = _safe_file(directory / "UNVERIFIED.onnx", run)
        _export(model, path, settings["constant_folding_requested"])
        settings.update(onnx_sha256=_hash_file(path), onnx_file=str(path),
                        onnx_opcounts=dict(Counter(node.op_type for node in onnx.load(path).graph.node)),
                        torch_batchnorm_modules=sum(isinstance(m, torch.nn.BatchNorm2d) for m in model.modules()))
        session = _session(path, profile)
        options = session.get_session_options()
        settings.update(providers=session.get_providers(), ort_intra_threads=options.intra_op_num_threads,
                        ort_inter_threads=options.inter_op_num_threads,
                        ort_optimization=str(options.graph_optimization_level), input_hashes={})

        def torch_outputs(x):
            with torch.no_grad():
                return [output.detach().cpu().numpy() for output in model(x)]

        def ort_outputs(x):
            return session.run(list(exporter.OUTPUT_NAMES), {exporter.INPUT_NAME: x.numpy()})

        def evaluate(name, x, *, source="random", mask=None, rtol=PARITY_RTOL, gate_case=True):
            return _record(directory, run, cases, name, lambda: torch_outputs(x), lambda: ort_outputs(x),
                           source=source, batch=len(x), mask=mask, rtol=rtol,
                           gate_case=gate_case, input_hash=_hash_array(x.numpy()))

        generator = torch.Generator().manual_seed(0)
        inputs = {size: torch.rand(size, 18, 8, 8, generator=generator) for size in (1, 7)}
        pair = None
        for size, x in inputs.items():
            name = f"random_{size}"
            settings["input_hashes"][name] = _hash_array(x.numpy())
            np.save(_safe_file(directory / f"{name}.npy", run), x.numpy(), allow_pickle=False)
            observed = evaluate(name, x)
            if size == 7:
                pair = observed
        row = inputs[7][failing_row:failing_row + 1]
        alone = evaluate(f"same_row_{failing_row}_cross", row, source="same_row", gate_case=False)
        if pair is not None and alone is not None:
            for index, runtime in enumerate(("torch", "ort")):
                expected = [head[failing_row:failing_row + 1] for head in pair[index]]
                _record(directory, run, cases, f"same_row_{failing_row}_{runtime}",
                        lambda expected=expected: expected, lambda index=index: alone[index],
                        source="same_row", batch=1, gate_case=False,
                        input_hash=_hash_array(row.numpy()))
        if positions_error is not None:
            def unavailable(error=positions_error):
                raise error
            _record(directory, run, cases, "heldout_load", unavailable, unavailable,
                    source="heldout", batch=0, rtol=0.0)
        else:
            planes, legal = positions["planes"], positions["legal_mask"]
            settings["heldout_positions"] = len(planes)
            for key, array in (("heldout_planes", planes.numpy()), ("heldout_legal_mask", legal.numpy())):
                settings["input_hashes"][key] = _hash_array(array)
                np.save(_safe_file(directory / f"{key}.npy", run), array, allow_pickle=False)
            for size in (1, 7, 32):
                for start in range(0, len(planes), size):
                    indices = torch.arange(start, start + size) % len(planes)
                    evaluate(f"heldout_{size}_{start}", planes[indices], source="heldout",
                             mask=legal[indices].numpy(), rtol=0.0)
        result["completed"] = all(case["completed"] for case in cases)
        result["gate_pass"] = result["completed"] and all(
            case["gate_pass"] for case in cases if case["gate_case"])
    except Exception as error:
        result["error"] = {"kind": "operational", "type": type(error).__name__, "message": str(error)}
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "evidence", "run", "report"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--profile", choices=PROFILES, required=True)
    parser.add_argument("--failing-row", type=int, default=2)
    args = parser.parse_args(argv)
    expected = args.evidence.resolve() / "profiles" / args.profile / "report.json"
    if args.report.resolve() != expected.resolve():
        parser.error("report must be evidence/profiles/PROFILE/report.json")
    report = run_profile(args.checkpoint, args.evidence, args.run, args.profile, args.failing_row)
    target = _safe_file(args.report, args.run.resolve())
    pending = _safe_file(target.with_name(target.name + ".tmp"), args.run.resolve())
    pending.write_text(json.dumps(report, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    pending.replace(target)
    print(f"{args.profile}: completed={report['completed']} gate_pass={report['gate_pass']}", flush=True)
    return 0 if report["completed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
