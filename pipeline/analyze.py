"""Export plot-ready RL history and a standalone HTML/SVG report (stdlib only)."""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
import sys

from analysis_data import ATTEMPT_COLUMNS, GEN_COLUMNS, STAGE_COLUMNS, SUMMARY_COLUMNS, TRAIN_COLUMNS, collect
from analysis_matches import COLUMNS as MATCH_COLUMNS, GAME_COLUMNS
from analysis_plot import render
from analysis_selfplay import COLUMNS as SELFPLAY_COLUMNS
from report import _publish

SCHEMAS = {"generations": GEN_COLUMNS, "training": TRAIN_COLUMNS, "matches": MATCH_COLUMNS,
           "training_attempts": ATTEMPT_COLUMNS,
           "match_games": GAME_COLUMNS, "selfplay": SELFPLAY_COLUMNS,
           "stages": STAGE_COLUMNS, "processes": STAGE_COLUMNS, "stage_summary": SUMMARY_COLUMNS}


def export_analysis(run_dir, out_dir=None) -> Path:
    """Read metadata only; export all generations, including live/rejected ones."""
    run = Path(run_dir).resolve(strict=True)
    out = Path(out_dir).resolve() if out_dir is not None else run / "analysis"
    # Analysis has fixed filenames and must not replace source artifact folders.
    reserved = [run, run / "gens", run / "supervised", run / "supervised_val"]
    if out in reserved or run / "gens" in out.parents or any(
            source in out.parents for source in reserved[2:]):
        raise ValueError("analysis output must be separate from run source artifacts")
    tables, warnings = collect(run)
    out.mkdir(parents=True, exist_ok=True)
    contents = []
    for name, columns in SCHEMAS.items():
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows({column: "NA" if row.get(column) is None else row[column] for column in columns}
                         for row in tables[name])
        contents.append((out / f"{name}.csv", stream.getvalue()))
    contents.extend(((out / "index.html", render(tables, warnings)),
                     (out / "diagnostics.json", json.dumps({"warnings": warnings}, indent=2) + "\n")))
    _publish(contents)
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    try:
        out = export_analysis(args.run, args.out)
        print(f"Saved analysis: {out / 'index.html'}")
        return 0
    except (OSError, ValueError, TypeError) as error:
        print(f"analysis: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
