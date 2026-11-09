"""Export a `ChessNet` checkpoint to ONNX and verify the SPEC.md §4 contract.

Usage: python training/export_onnx.py --checkpoint model.pt --out model.onnx
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from checkpoint import load_checkpoint
from models.net import IN_PLANES, NUM_MOVES, NUM_WDL, ChessNet
from onnx_parity import PARITY_ATOL, PARITY_RTOL, PROBABILITY_ATOL, compare_outputs

OPSET = 17
INPUT_NAME = "input_planes"
OUTPUT_NAMES = ("policy_logits", "wdl_logits")


def export_onnx(model: ChessNet, out_path: str | Path) -> Path:
    """Export `model` (eval, CPU) to a single-file ONNX with dynamic batch size.

    Uses the TorchScript exporter (`dynamo=False`): the dynamo exporter in
    torch 2.15 only emits opset >= 18 and fails to down-convert to SPEC's 17.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model = model.eval().cpu()
    dummy = torch.zeros(2, IN_PLANES, 8, 8)
    axes = {0: "batch_size"}
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*legacy TorchScript-based ONNX export.*")
        warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"torch\.onnx")
        torch.onnx.export(
            model,
            (dummy,),
            str(out_path),
            input_names=[INPUT_NAME],
            output_names=list(OUTPUT_NAMES),
            opset_version=OPSET,
            dynamic_axes={name: axes for name in (INPUT_NAME, *OUTPUT_NAMES)},
            dynamo=False,
        )
    onnx.checker.check_model(str(out_path))
    return out_path


def _shape(value_info: onnx.ValueInfoProto) -> list[int | str]:
    dims = value_info.type.tensor_type.shape.dim
    return [d.dim_param if d.dim_param else d.dim_value for d in dims]


def check_contract(path: str | Path) -> None:
    """Raise AssertionError unless the ONNX file matches SPEC.md §4."""
    model = onnx.load(str(path))
    opsets = {o.domain: o.version for o in model.opset_import}
    assert opsets.get("", opsets.get("ai.onnx")) == OPSET, f"opset {opsets}, want {OPSET}"
    float_t = onnx.TensorProto.FLOAT
    expected_out = {OUTPUT_NAMES[0]: NUM_MOVES, OUTPUT_NAMES[1]: NUM_WDL}

    (inp,) = model.graph.input
    assert inp.name == INPUT_NAME, inp.name
    assert inp.type.tensor_type.elem_type == float_t
    in_shape = _shape(inp)
    assert isinstance(in_shape[0], str) and in_shape[0], f"input dim 0 not dynamic: {in_shape}"
    assert in_shape[1:] == [IN_PLANES, 8, 8], in_shape

    assert [o.name for o in model.graph.output] == list(OUTPUT_NAMES)
    for out in model.graph.output:
        assert out.type.tensor_type.elem_type == float_t, out.name
        shape = _shape(out)
        assert isinstance(shape[0], str) and shape[0], f"{out.name} dim 0 not dynamic: {shape}"
        assert shape[1:] == [expected_out[out.name]], f"{out.name}: {shape}"


def verify_parity(
    model: ChessNet, path: str | Path, inputs: torch.Tensor, *,
    legal_mask: torch.Tensor | None = None, rtol: float = PARITY_RTOL, session=None,
) -> float:
    """Check logits and probabilities; return the largest raw absolute error.

    Large random-input logits use a per-value relative allowance. Real chess
    inputs pass rtol=0 to retain the strict absolute 1e-4 bound.
    """
    model = model.eval().cpu()
    x = inputs.detach().cpu().float()
    with torch.no_grad():
        ref = [t.numpy() for t in model(x)]
    sess = session if session is not None else ort.InferenceSession(
        str(path), providers=["CPUExecutionProvider"]
    )
    got = sess.run(list(OUTPUT_NAMES), {INPUT_NAME: x.numpy()})
    mask = None if legal_mask is None else legal_mask.detach().cpu().numpy()
    return compare_outputs(ref, got, mask, rtol=rtol)


def load_parity_positions(source: Path) -> dict[str, torch.Tensor]:
    """Read at most 64 evenly spaced Rust-written samples through memmaps."""
    from data.chunk_dataset import ChunkDataset, list_complete_chunks

    paths = list_complete_chunks(source) if source.is_dir() else [source]
    dataset = ChunkDataset(paths)
    if not len(dataset):
        raise ValueError(f"{source}: parity source has no complete nonempty chunks")
    indices = np.linspace(0, len(dataset) - 1, min(64, len(dataset)), dtype=np.int64)
    samples = [dataset[int(index)] for index in indices]
    return {key: torch.stack([sample[key] for sample in samples])
            for key in ("planes", "legal_mask")}


def verify_real_positions(
    model: ChessNet, path: Path, source: Path, session, *, positions=None,
) -> float:
    positions = load_parity_positions(source) if positions is None else positions
    planes, legal = positions["planes"], positions["legal_mask"]
    worst = 0.0
    for size in (1, 7, 32):
        batch_worst = 0.0
        for start in range(0, len(planes), size):
            # Repeat selected positions to exercise the requested batch dimension
            # even for tiny validation sets and partial final batches.
            indices = torch.arange(start, start + size) % len(planes)
            diff = verify_parity(
                model, path, planes[indices],
                legal_mask=legal[indices], rtol=0.0, session=session,
            )
            batch_worst = max(batch_worst, diff)
        print(f"parity real positions={len(planes)} batch={size}: "
              f"max raw diff {batch_worst:.2e}; absolute limit {PARITY_ATOL:g}", flush=True)
        worst = max(worst, batch_worst)
    return worst


def validate_paths(checkpoint: Path, out: Path, source: Path | None) -> None:
    protected = [checkpoint]
    if source is not None:
        if source.is_dir():
            if out.resolve().is_relative_to(source.resolve()):
                raise ValueError("export output must be outside the parity source directory")
            protected.extend(path for path in source.iterdir() if path.is_file())
        else:
            protected.extend((source, source.with_suffix(".json")))
    for path in protected:
        if out.resolve() == path.resolve() or (
            out.exists() and path.exists() and out.samefile(path)
        ):
            raise ValueError(f"export output aliases protected input: {path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--parity-source", type=Path,
                        help="Rust chunk .bin or directory for strict real-position checks")
    args = parser.parse_args(argv)

    try:
        validate_paths(args.checkpoint, args.out, args.parity_source)
        positions = (load_parity_positions(args.parity_source)
                     if args.parity_source is not None else None)
        model = load_checkpoint(args.checkpoint)
        export_onnx(model, args.out)
        check_contract(args.out)
        session = ort.InferenceSession(str(args.out), providers=["CPUExecutionProvider"])
        gen = torch.Generator().manual_seed(0)
        worst = 0.0
        for batch in (1, 7):
            x = torch.rand(batch, IN_PLANES, 8, 8, generator=gen)
            diff = verify_parity(model, args.out, x, session=session)
            worst = max(worst, diff)
            print(f"parity random batch={batch}: max raw diff {diff:.2e}; "
                  f"per-value bound {PARITY_ATOL:g} + {PARITY_RTOL:g} * abs(reference)",
                  flush=True)
        if args.parity_source is not None:
            worst = max(worst, verify_real_positions(
                model, args.out, args.parity_source, session, positions=positions,
            ))
    except (AssertionError, ValueError, OSError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1
    print(f"exported {args.out} (opset {OPSET}); parity passed, max raw diff {worst:.2e}; "
          f"probability limit {PROBABILITY_ATOL:g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
