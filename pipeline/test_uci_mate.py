#!/usr/bin/env python3
"""GATE 4: the compiled engine finds mate-in-1 over UCI with `go nodes 400`.

Standard library only, so it runs with any `python3`. Runs the engine with no
`--model`, i.e. the uniform dummy network: the mate must be found by search.
Expected sets list *every* mating move (verified with python-chess).
"""

from __future__ import annotations

import queue
import subprocess
import sys
import threading
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ENGINE_DIR = REPO_ROOT / "engine"
ENGINE_BIN = ENGINE_DIR / "target" / "release" / "engine"
TIMEOUT_S = 60.0
NODES = 400

MATE_IN_ONE = [
    ("scholar (white)", "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 4 4", {"f3f7"}),
    ("back rank (black)", "r5k1/8/8/8/8/8/5PPP/6K1 b - - 0 1", {"a8a1"}),
    ("promotion (white)", "k7/2P5/1K6/8/8/8/8/8 w - - 0 1", {"c7c8q", "c7c8r"}),
]


class Engine:
    """Minimal UCI driver with a reader thread so reads can time out."""

    def __init__(self, path: Path):
        self.proc = subprocess.Popen(
            [str(path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self.lines: queue.Queue[str | None] = queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self) -> None:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            self.lines.put(line.rstrip("\n"))
        self.lines.put(None)

    def send(self, cmd: str) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()

    def read_until(self, prefix: str) -> list[str]:
        """All lines up to and including the first starting with `prefix`."""
        seen: list[str] = []
        while True:
            try:
                line = self.lines.get(timeout=TIMEOUT_S)
            except queue.Empty:
                raise TimeoutError(f"no {prefix!r} within {TIMEOUT_S}s; got {seen}") from None
            if line is None:
                raise RuntimeError(f"engine exited before {prefix!r}; got {seen}")
            seen.append(line)
            if line.startswith(prefix):
                return seen

    def close(self) -> None:
        try:
            self.send("quit")
            self.proc.wait(timeout=10)
        except Exception:
            self.proc.kill()


def main() -> int:
    subprocess.run(
        ["cargo", "build", "--release", "--manifest-path", str(ENGINE_DIR / "Cargo.toml")],
        check=True,
    )
    engine = Engine(ENGINE_BIN)
    failures = 0
    try:
        engine.send("uci")
        engine.read_until("uciok")
        engine.send("isready")
        engine.read_until("readyok")
        for name, fen, mates in MATE_IN_ONE:
            engine.send("ucinewgame")
            engine.send(f"position fen {fen}")
            engine.send(f"go nodes {NODES}")
            lines = engine.read_until("bestmove")
            best = lines[-1].split()[1]
            info = next((l for l in reversed(lines) if l.startswith("info nodes")), "")
            ok = best in mates
            failures += not ok
            status = "PASS" if ok else "FAIL"
            print(f"{status}  {name:<18} bestmove {best:<6} want {sorted(mates)}  | {info}")
            for l in lines:
                if l.startswith("info string"):
                    print(f"      {l}")
    finally:
        engine.close()
    print(f"{len(MATE_IN_ONE) - failures}/{len(MATE_IN_ONE)} mate-in-1 positions solved")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
