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

OPSET = 17
INPUT_NAME = "input_planes"
OUTPUT_NAMES = ("policy_logits", "wdl_logits")
PARITY_ATOL = 1e-4


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


def verify_parity(model: ChessNet, path: str | Path, inputs: torch.Tensor) -> float:
    """Max |PyTorch - onnxruntime| over both outputs for `inputs` (CPU, eval)."""
    model = model.eval().cpu()
    x = inputs.detach().cpu().float()
    with torch.no_grad():
        ref = [t.numpy() for t in model(x)]
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    got = sess.run(list(OUTPUT_NAMES), {INPUT_NAME: x.numpy()})
    diffs = []
    for r, g in zip(ref, got):
        assert r.shape == g.shape, (r.shape, g.shape)
        diffs.append(float(np.max(np.abs(r - g))))
    return max(diffs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)

    model = load_checkpoint(args.checkpoint)
    export_onnx(model, args.out)
    check_contract(args.out)
    gen = torch.Generator().manual_seed(0)
    worst = 0.0
    for batch in (1, 7):
        x = torch.rand(batch, IN_PLANES, 8, 8, generator=gen)
        worst = max(worst, verify_parity(model, args.out, x))
    print(f"exported {args.out} (opset {OPSET}); max parity diff {worst:.2e}")
    if worst > PARITY_ATOL:
        print(f"FAIL: parity diff {worst:.2e} > {PARITY_ATOL:.0e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
