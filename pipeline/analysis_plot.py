"""Dependency-free, standalone SVG plots for saved RL diagnostics."""

from __future__ import annotations

from html import escape
import math

COLORS = ("#2563eb", "#dc2626", "#059669", "#9333ea", "#d97706", "#0891b2")


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _chart(title, xlabel, ylabel, series, limits=None):
    series = [(name, [point for point in points if all(_finite(v) for v in point[:2])])
              for name, points in series]
    series = [(name, points) for name, points in series if points]
    if not series:
        return f"<section><h2>{escape(title)}</h2><p>No measurements available yet.</p></section>"
    xs = [point[0] for _, points in series for point in points]
    ys = [v for _, points in series for point in points for v in point[1:] if _finite(v)]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = limits if limits else (min(ys), max(ys))
    if xmin == xmax:
        xmin, xmax = xmin - 0.5, xmax + 0.5
    if ymin == ymax:
        ymin, ymax = ymin - 0.5, ymax + 0.5
    elif not limits:
        margin = (ymax - ymin) * 0.08
        ymin, ymax = ymin - margin, ymax + margin
    sx = lambda x: 80 + (x - xmin) / (xmax - xmin) * 770
    sy = lambda y: 285 - (y - ymin) / (ymax - ymin) * 235
    parts = [f'<section><h2>{escape(title)}</h2><svg role="img" aria-label="{escape(title)}" '
             'viewBox="0 0 920 355"><rect width="920" height="355" fill="white"/>']
    for tick in range(6):
        x, y = xmin + (xmax - xmin) * tick / 5, ymin + (ymax - ymin) * tick / 5
        parts.extend((f'<line x1="80" y1="{sy(y):.2f}" x2="850" y2="{sy(y):.2f}" stroke="#e2e8f0"/>',
                      f'<text x="70" y="{sy(y)+4:.2f}" text-anchor="end">{y:.4g}</text>',
                      f'<text x="{sx(x):.2f}" y="307" text-anchor="middle">{x:.4g}</text>'))
    parts.extend(('<path d="M80 50V285H850" fill="none" stroke="#475569"/>',
                  f'<text x="460" y="339" text-anchor="middle">{escape(xlabel)}</text>',
                  f'<text transform="translate(18,175) rotate(-90)" text-anchor="middle">{escape(ylabel)}</text>'))
    legends = []
    for index, (name, points) in enumerate(series):
        color = COLORS[index % len(COLORS)]
        coords = " ".join(f"{sx(p[0]):.2f},{sy(p[1]):.2f}" for p in points)
        parts.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2"/>')
        for point in points:
            if len(point) >= 4 and all(_finite(v) for v in point[2:4]):
                parts.append(f'<line x1="{sx(point[0]):.2f}" x2="{sx(point[0]):.2f}" '
                             f'y1="{sy(point[2]):.2f}" y2="{sy(point[3]):.2f}" stroke="{color}"/>')
            parts.append(f'<circle cx="{sx(point[0]):.2f}" cy="{sy(point[1]):.2f}" r="3" fill="{color}">'
                         f'<title>{escape(name)}: {point[0]:.5g}, {point[1]:.5g}</title></circle>')
        legends.append(f'<span><i style="background:{color}"></i>{escape(name)}</span>')
    return "".join(parts) + '</svg><div class="legend">' + "".join(legends) + "</div></section>"


def _series(rows, x, fields):
    return [(label, [(row.get(x), row.get(key)) for row in rows]) for key, label in fields]


def _training(rows, fields):
    groups = {}
    for row in rows:
        key = (row["gen"], row.get("attempt_id"), row.get("attempt_status"))
        groups.setdefault(key, []).append(row)
    series = []
    for (gen, attempt, status), records in groups.items():
        for key, label in fields:
            name = f"gen {gen} / {str(attempt)[:8]} ({status}) / {label}"
            series.append((name, [(row.get("steps"), row.get(key)) for row in records]))
    return series


def render(tables, warnings):
    generations, training = tables["generations"], tables["training"]
    charts = [
        _chart("Training objective", "Optimizer step (each attempt starts at zero)", "Block mean loss",
               _training(training, [("train_loss", "Q-blended optimization objective")])),
        _chart("Fixed-probe and held-out policy loss", "Optimizer step", "Policy cross-entropy (saved targets)",
               _training(training, [("train_probe_policy_loss", "training probe"), ("val_policy_loss", "held-out")])),
        _chart("Fixed-probe and held-out WDL loss", "Optimizer step", "Hard-label WDL cross-entropy",
               _training(training, [("train_probe_wdl_loss", "training probe"), ("val_wdl_loss", "held-out")])),
        _chart("Fixed-probe and held-out policy accuracy", "Optimizer step", "Policy top-1 fraction",
               _training(training, [("train_probe_policy_top1", "training probe"), ("val_policy_top1", "held-out")]), (0, 1)),
        _chart("Held-out WDL probability quality", "Optimizer step", "WDL Brier score (lower is better)",
               _training(training, [("val_wdl_brier", "held-out WDL Brier")])),
        _chart("Candidate validation by generation", "Generation", "Policy top-1 fraction",
               _series(generations, "gen", [("val_policy_top1", "fixed supervised holdout")]), (0, 1)),
        _chart("Baseline match scores", "Generation", "Score fraction",
               _series(generations, "gen", [("vs_random", "vs random"), ("vs_material", "vs material")]), (0, 1)),
        _chart("Candidate relative strength", "Generation", "Elo vs that generation's incumbent (95% CI)",
               [("candidate vs incumbent", [(row["gen"], row.get("elo"), row.get("elo_lo"), row.get("elo_hi"))
                                            for row in generations])]),
        _chart("Promoted generation ladder", "Generation", "Relative ladder Elo (gen 0 = 0)",
               _series(generations, "gen", [("ladder_elo", "promoted generation")]))]
    groups = {}
    for row in tables["matches"]:
        groups.setdefault(row["gen"], []).append(row)
    likelihood = []
    for gen, records in groups.items():
        likelihood.extend((f"gen {gen}: {label}", [(row["games"], row.get(key)) for row in records])
                          for key, label in (("llr", "LLR"), ("sprt_lower", "reject bound"), ("sprt_upper", "accept bound")))
    charts.extend((
        _chart("Promotion evidence", "Cumulative match games", "SPRT log-likelihood ratio", likelihood),
        _chart("Self-play outcomes", "Generation", "Fraction", _series(generations, "gen", [
            ("selfplay_draw_rate", "draw rate"), ("selfplay_false_positive_rate", "resignation false-positive rate")]), (0, 1)),
        _chart("Stage wall time", "Generation", "Seconds", _series(generations, "gen", [
            ("selfplay_seconds", "self-play"), ("train_seconds", "training"), ("match_seconds", "matches")])),
        _chart("Largest measured process RSS", "Generation", "Bytes", _series(generations, "gen", [
            ("peak_rss_bytes", "measured process maximum (see completeness in CSV)")]))))
    links = " ".join(f'<a href="{name}.csv">{name}.csv</a>' for name in tables)
    diagnostics = "".join(f"<li>{escape(item)}</li>" for item in warnings)
    return ('<!doctype html><html lang="en"><meta charset="utf-8"><title>Chessbot run analysis</title>'
            '<style>body{font:15px system-ui;color:#172033;max-width:1100px;margin:32px auto;padding:0 20px;}'
            'section{border:1px solid #dbe2ec;border-radius:8px;padding:16px;margin:24px 0;}svg{width:100%;}'
            'svg text{font:13px system-ui;fill:#475569}.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:12px;}'
            '.legend i{display:inline-block;width:12px;height:3px;margin:0 6px 3px 0;}a{margin-right:12px;}</style>'
            '<h1>Chessbot run analysis</h1><p>Each training attempt is a separate curve. '
            'Q-blended WDL optimization loss uses a different target from outcome-only probe/validation WDL loss. '
            'Policy evaluation retains the saved policy targets, including soft self-play targets. '
            'A falling training-probe loss with worsening held-out loss is a warning worth investigating; '
            'the supervised holdout and changing RL replay distribution can also differ. '
            'Compare probe trends within an attempt; probes can change across generations. '
            'Match evidence measures playing strength. Missing values are unavailable, never zero.</p>'
            '<p>Candidate Elo compares each candidate with its own incumbent. Ladder Elo is relative, '
            'not a Stockfish calibration. Confidence intervals describe individual matches.</p>'
            f'<nav>{links}</nav>' + "".join(charts) +
            f'<section><h2>Input diagnostics ({len(warnings)})</h2><ul>{diagnostics}</ul></section></html>')
