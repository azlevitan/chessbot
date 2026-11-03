# SPEC.md — Technical Contracts & Data Schemas

## 1. Canonical Perspective & Move Indexing (`4,168` Action Space)
All boards and moves are canonicalized relative to the **side to move** (`us` vs `them`).
- If `side_to_move == Color::Black`, mirror every square vertically (`sq_idx ^ 56`) before encoding planes or move indices.
- **Deterministic Move Bijection (`0..4168`):**
  Given canonical `from_sq` (`0..63`) and `to_sq` (`0..63`):
  1. For all non-promotions AND Queen promotions (`Promotion::None` or `Promotion::Queen`):
     `index = from_sq * 64 + to_sq` (indices `0..4095`).
  2. For underpromotions (`Knight = 0`, `Bishop = 1`, `Rook = 2`), where a pawn moves from rank 6 (`from_sq` in `48..55`):
     - `from_file = from_sq % 8` (`0..7`)
     - `delta = (to_sq % 8) as i32 - (from_file as i32) + 1` (`0` for left capture, `1` for straight, `2` for right capture)
     - `index = 4096 + (from_file * 9) + (delta as usize * 3) + promo_idx` (indices `4096..4167`).
- **Castling** is indexed by the king's destination square (`e1g1` / `e1c1` canonical), matching UCI and python-chess — NOT cozy-chess's internal king-captures-rook form. The conversion lives only in `move_mapper.rs`.
- **Decoding needs the board:** a Queen promotion shares `from*64+to` with no other legal move, but its index is indistinguishable from a non-promotion without context. `index_to_move(&Board, idx)` attaches `Promotion::Queen` when the piece on `from` is a pawn landing on the last rank.
- **Illegal Move Masking:** Both in MCTS and loss computation, logits for illegal moves MUST be masked to `-1e9` prior to Softmax.

## 2. Input Tensor Schema (`18 x 8 x 8` Float32 Planes)
Constructed in Rust from a `18 * 64` flat array:
- Planes `0..=5`: `us` pieces (`Pawn, Knight, Bishop, Rook, Queen, King`). `1.0` if present on square, else `0.0`.
- Planes `6..=11`: `them` pieces (`Pawn, Knight, Bishop, Rook, Queen, King`), already vertically flipped if Black is to move.
- Plane `12`: `us` kingside castling right (all `1.0` if the *right* is held — not whether castling is legal this move — else `0.0`).
- Plane `13`: `us` queenside castling right.
- Plane `14`: `them` kingside castling right.
- Plane `15`: `them` queenside castling right.
- Plane `16`: En passant target square (`1.0` on the canonical target square if available, else `0.0`).
- Plane `17`: Halfmove (50-move rule) clock normalized as `halfmove_clock as f32 / 100.0`.

## 3. Shared Binary Training Sample (`#[repr(C, packed)]`)
Written by Rust (`engine`) and read via `np.memmap` in Python (`training`). Total size: **492 bytes per sample** (little-endian).

~~~rust
#[repr(C, packed)]
pub struct SparsePolicyEntry {
    pub move_idx: u16, // 0..4167 (u16::MAX = unused padding slot)
    pub prob: f32,     // MCTS visit probability (or 1.0 for supervised played move)
}

#[repr(C, packed)]
pub struct TrainingSample {
    pub bitboards: [u64; 12],             // 96 bytes: Planes 0..11 packed as 64-bit integers
    pub castling_and_ep: u8,              // 1 byte: bit0 = us K, bit1 = us Q, bit2 = them K, bit3 = them Q, bit4 = has_ep
    pub ep_square: u8,                    // 1 byte: canonical ep target square (40..47 when has_ep, else 0)
    pub halfmove_clock: u8,               // 1 byte: clamped to 0..100
    pub outcome_wdl: u8,                  // 1 byte: 0 = Loss, 1 = Draw, 2 = Win (from `us` perspective)
    pub q_value: f32,                     // 4 bytes: MCTS root evaluation in [-1.0, 1.0]
    pub policy: [SparsePolicyEntry; 64],  // 384 bytes: legal moves & probabilities
    pub num_legal_moves: u32,             // 4 bytes: Number of active entries in `policy`
}
~~~

**Policy slot rules:**
- Every legal move is written (prob `0.0` for unvisited / unplayed moves), so the active entries are the exact legal-move mask for the loss.
- Unused slots are `{ move_idx: u16::MAX, prob: 0.0 }`.
- If a position has more than 64 legal moves, keep the top 64 by prob (ties → lower `move_idx`) and renormalize the kept probs to sum to 1. `num_legal_moves` is then 64 (entries written), not the true legal count.
- Active entries are written sorted by ascending `move_idx`.

*Note for Python reader:* Unpack the 12 `uint64` bitboards into `(12, 8, 8)` `float32` planes using bitwise shifts `((bb >> sq) & 1)` inside `__getitem__` so disk storage stays compact.

## 4. Neural Network & ONNX Contract
- **Architecture:** `SE-ResNet-6x64`
  - Stem: `Conv2d(18, 64, kernel_size=3, padding=1, bias=False)` -> `BatchNorm2d(64)` -> `ReLU`
  - Trunk: 6 Residual Blocks, each with two `3x3` conv layers (64 channels) + Squeeze-and-Excitation (reduction ratio 4).
  - Policy Head: `Conv2d(64, 32, kernel_size=1)` -> `BN` -> `ReLU` -> `Flatten` -> `Linear(32*64, 4168)` outputting `policy_logits`.
  - Value (WDL) Head: `Conv2d(64, 8, kernel_size=1)` -> `BN` -> `ReLU` -> `Flatten` -> `Linear(8*64, 64)` -> `ReLU` -> `Linear(64, 3)` outputting `wdl_logits` (`[Loss, Draw, Win]`).
- **ONNX Export Contract:**
  - Opset: `17`
  - Input name: `"input_planes"`, shape `[batch_size, 18, 8, 8]`, type `float32`
  - Output 0 name: `"policy_logits"`, shape `[batch_size, 4168]`, type `float32`
  - Output 1 name: `"wdl_logits"`, shape `[batch_size, 3]`, type `float32`
  - Dynamic axis `0` (`batch_size`) on all inputs and outputs.
- **Rust `ort` (`=2.0.0-rc.15`) Session Initialization:**
  - Explicitly initialize the environment first via `let env = ort::init().build()?;` and pass `&env` into `Session::builder(&env)?` with the CoreML execution provider enabled.

## 5. MCTS Search Formula (PUCT)
- Scalar value from `wdl_logits`: after Softmax `[p_loss, p_draw, p_win]`, compute `v = p_win - p_loss` in `[-1.0, 1.0]`.
- At node `s`, pick legal child action `a` maximizing:
  `Q(s, a) + c_puct * P(s, a) * sqrt(N(s)) / (1 + N(s, a))`
  where `c_puct = 1.5`, and `Q(s, a)` is negated when backing up between alternating sides to move.
- **First Play Urgency (FPU, Task 7.1):** For an unvisited child, use `Q = 0.0` at the root and `Q = (-parent_edge_q - 0.25).clamp(-1.0, 1.0)` elsewhere. The edge into the current node is scored from its parent's side to move, so negation converts it to the current side's perspective. Visited children use their measured Q.

## 6. Sample Chunks & Sidecars
A **chunk** is a plain `.bin` file of back-to-back §3 `TrainingSample` records — no header, so §3 readers (`open_samples`, `ReplayBuffer`) work unchanged. Metadata lives in a JSON **sidecar** with the same stem (`chunk_000042.bin` ↔ `chunk_000042.json`).
- **Naming:** `chunk_{index:06}.bin` inside a chunk directory; `index` starts at `0` and is dense.
- **Atomic write order (Rust writer):** write `<stem>.bin.tmp` → flush + `sync_all` → rename to `<stem>.bin` → write `<stem>.json.tmp` → rename to `<stem>.json`.
- **A chunk is complete iff** `<stem>.json` exists AND `size(<stem>.bin) == json.samples * 492`. Readers MUST ignore incomplete chunks and every `*.tmp` file. A writer about to produce index `i` first deletes any stale `chunk_i.bin`, `chunk_i.bin.tmp`, `chunk_i.json.tmp`.
- **Sidecar schema** (`format` is always `"chessbot-chunk-v1"`; extra keys allowed, readers ignore them):
~~~json
{ "format": "chessbot-chunk-v1", "kind": "selfplay", "samples": 7421, "games": 50,
  "model": "data_store/runs/main/gens/gen_0003/model.onnx",   // or "uniform"
  "seed": 3001000, "sims": 200, "max_plies": 512, "temp_plies": 30,
  "results": { "white": 12, "black": 10, "draw": 28 },
  "terminations": { "checkmate": 18, "resign": 4, "rule_draw": 3, "repetition": 9,
                    "insufficient_material": 6, "ply_cap": 10 },
  "resign": { "threshold": -0.95, "consecutive": 3, "playthrough_games": 5, "false_positives": 0 } }
~~~
~~~json
{ "format": "chessbot-chunk-v1", "kind": "pgn", "samples": 262144, "games": 3120,
  "source": "lichess_elite_2025-01.pgn", "first_game": 2000 }   // "source" is "-" for stdin
~~~
- Selfplay game `i` (0-based within one `engine selfplay` call) uses RNG seed `seed + i`. The loop picks `seed = 1_000_000 * gen + 1_000 * chunk_index + config.seed` (§10), so `games_per_chunk <= 1000`.
- Games are written in game-index order (a finished game waits in memory until all lower-index games are written), so output bytes do not depend on `--parallel`.

## 7. Self-Play Policy
- **Noise:** Root Dirichlet `alpha = 0.3`, `eps = 0.25`, mixed into the root children's priors once per move: right after the root is expanded, or right away if the root is already expanded because of tree reuse.
- **Simulations:** `--sims N` means `N` *new* playouts per move. With tree reuse, the root may already hold visits. The policy target is the visit share over **all** root child visits (reused + new), written for every legal move (0.0 if unvisited). §3 slot rules apply.
- **Temperature schedule:** For plies `0 .. temp_plies-1` (default `30`), the move is sampled in proportion to root child visits (τ = 1). From ply `temp_plies` on, the most-visited child is played (ties → higher prior, then lower canonical move index).
- **Tree reuse:** After a move is played, the chosen child's subtree is compacted into a fresh arena (still ≤ 65,536 nodes) and becomes the next root. The noise from the earlier move is not carried forward: non-root priors are always clean network priors.
- **Termination:** checkmate; stalemate / 50-move (`GameStatus::Drawn`); threefold repetition (by `Board::hash()` over the game history); insufficient material (K v K, K+minor v K); `max_plies` (default `512`) → draw; resignation (below).
- **Resignation:** A side resigns when its own `root_q < resign_threshold` on `resign_consecutive` (default `3`) of its own moves in a row. Resignation is disabled when `threshold <= -1.0`. Each game draws one `f32` from its RNG at game start (before any search); if it is `< resign_playthrough` (default `0.1`), the game is a *playthrough* game: it never resigns, but records the first side that would have resigned. A playthrough game where that side did **not** lose counts as a *false positive*. Labels always use the real final result. A resigned game is a loss for the resigner (`outcome_wdl` from each side's perspective, as in §3).
- **Outcome back-fill:** Every sample of a game gets `outcome_wdl` from the final result relative to that sample's side to move, and `q_value` = that ply's MCTS `root_q`.

## 8. Batched Search & Benchmark
- **Batch size `B`** (`SearchLimits.batch`, default `1`). `B = 1` MUST reproduce the pre-batching search exactly (same visits for the same evaluator).
- **Visit accounting:** Initial root expansion counts as one simulation, preserving Task 7.1's golden baseline. On a fresh, expanded tree, `root.visits == simulations` and every expanded node (root included) has `visits == 1 + sum(child.visits)`. Thus 400 simulations give 399 root-child visits. Pending virtual losses are excluded from completed simulations and fully removed before returning a result.
- **Virtual loss:** When a leaf is selected for evaluation, every node on its path (root excluded) gets `visits += 1` and `value_sum -= 1.0` (a loss from the perspective of the side that chose that node). When the batch's results arrive, each path first undoes its virtual loss and then applies the normal §5 alternating backup.
- **Gathering:** Up to `min(B, remaining sims)` leaves are gathered per evaluator call. Terminal leaves are backed up at once and count as simulations. A path that reaches a leaf already pending in this batch is a *collision*: undo its virtual loss and stop gathering for this batch.
- **Multi-game batching (selfplay/match):** Each of `P` concurrent games contributes its pending leaves to **one** evaluator call per step. Each game's tree has its own arena (≤ 65,536 nodes ≈ 1.8 MB).
- **Inference options:** `--ep auto|cpu|coreml` (`auto` = CoreML on macOS, CPU elsewhere; CoreML is never registered on non-macOS targets) and `--threads T` (ORT intra-op threads, default `1`).
- **`engine bench` output:** one JSON line on stdout:
  `{"sims":4800,"evals":4710,"seconds":0.41,"sims_per_sec":11707.3,"evals_per_sec":11487.8,"avg_batch":7.6,"ep":"uniform"}`

## 9. Matches & Evaluation
- **CLI:** `engine match --a <PLAYER> --b <PLAYER> --games <N even> --sims <S> --seed <X> --out <match.json> [--openings <file>] [--opening-offset K] [--max-plies 300] [--batch 8] [--moves-out <txt>] [--ep ..] [--threads ..]`
- **`<PLAYER>`:** a path to a `.onnx` file | `uniform` (MCTS + uniform net) | `material` (MCTS + material evaluator) | `random` (uniformly random legal move, RNG seeded `X + game`).
- **Material evaluator:** uniform policy. It counts us/them material from planes 0..11 (P=1, N=3, B=3, R=5, Q=9, K=0), sets `d = us - them` and `v = tanh(0.25 * d)`, and reports WDL logits `ln(max(p, 1e-6))` with `p_win = (1+v)/2`, `p_draw = 0`, `p_loss = (1-v)/2`.
- **Openings:** `engine/data/openings.txt` holds 50 lines. Each line is a space-separated UCI move sequence of 2–8 plies from the start position. Games `2k` and `2k+1` both use opening `(K + k) mod 50`. In game `2k`, A plays White; in game `2k+1`, B plays White.
- **Play:** no Dirichlet noise, most-visited move, tree reuse on, a separate tree per player. Same termination rules as §7 minus resignation. At `max_plies`, the game is **adjudicated by material**: `|d| >= 5` → win for the side ahead, else draw.
- **`match.json`:**
~~~json
{ "format": "chessbot-match-v1", "a": "gens/gen_0004/model.onnx", "b": "gens/gen_0003/model.onnx",
  "games": 40, "a_wins": 15, "b_wins": 9, "draws": 16, "a_score": 0.575,
  "sims": 200, "seed": 4000, "opening_offset": 20, "max_plies": 300,
  "results": [ { "game": 0, "opening": 20, "white": "a", "result": "1-0",
                 "termination": "checkmate", "plies": 87 } ] }
~~~
- **`--moves-out`** (optional): one line per game, `<game> <white:a|b> <result> <uci moves…>`, opening moves included.
- **Elo from a score:** `x = (W + 0.5 D) / n`, clamped to `[1/(2n), 1 - 1/(2n)]`; `elo = -400 * log10(1/x - 1)`. 95% CI: `x ± 1.96 * sqrt(var / n)` mapped through the same formula, where `var = (W(1-x)² + D(0.5-x)² + L x²) / n`.
- **SPRT (GSPRT approximation):** `s_i = 1 / (1 + 10^(-elo_i/400))` for `elo0`, `elo1`; `LLR = n * (s1 - s0) * (2x - s0 - s1) / (2 * max(var, 1e-3))`. Bounds are `lower = ln(β/(1-α))` and `upper = ln((1-β)/α)` (±2.944 for α = β = 0.05).

## 10. Run Directory, RL Loop & Promotion
~~~
<run>/                                   e.g. data_store/runs/main
  run.json                               loop state (atomic tmp+rename writes)
  loop.lock                              PID of the running loop
  supervised/chunk_NNNNNN.{bin,json}     convert-pgn output (training)
  supervised_val/chunk_000000.{bin,json} first `val_games` PGN games (held out)
  gens/gen_GGGG/                         4-digit zero-padded generation
    selfplay/chunk_NNNNNN.{bin,json}     made by the best net at the start of gen G (absent for G = 0)
    model.pt  model.onnx  train.json     candidate G
    match/block_BB.json                  candidate G vs best (§9 match.json per block)
    decision.json                        promotion decision
    anchors/vs_random.json  anchors/vs_material.json   (promoted gens only)
  ladder.json  report.tsv                derived; rebuilt by pipeline/report.py
~~~
- **`run.json`:** `{"format":"chessbot-run-v1","config":{…},"best_gen":3,"current_gen":4,"stage":"selfplay","resign_threshold":-0.95}`. `stage` is one of `selfplay → train → export → match → decide → anchors → prune → done`. After `done`, `current_gen += 1` and `stage = "selfplay"`.
- **Generation 0** is the supervised net (`pipeline/bootstrap.py`). It is always promoted (`decision.json` has `"reason": "bootstrap"`).
- **Stage idempotence (resume rule):** Every stage checks its outputs before doing any work. selfplay: chunks `0..ceil(games/games_per_chunk)-1` complete (§6). train: `model.pt` and `train.json` exist. export: `model.onnx` exists and passes `check_contract`. match: one block file per block played, then stop at the SPRT decision or `max_games`. decide: `decision.json` exists. Partial outputs (`*.tmp`, a `.pt` without `train.json`) are deleted and redone.
- **Training a candidate:** Start from the best gen's `model.pt` with a fresh AdamW optimizer. Train on the **replay window**:
  - Self-play chunks from gens `max(1, G-W+1) .. G`, taking whole chunks newest-first until the total reaches `window.max_samples` (default `W = 4`, `max_samples = 1,000,000` ≈ 492 MB on disk, memmapped, never loaded into RAM).
  - Plus the `supervised/` chunks, sampled with batch fraction `window.supervised_weight[min(G-1, len-1)]` (default `[0.5, 0.3, 0.2, 0.1, 0.0]`).
  - Each draw picks a source by weight, then a sample uniformly within that source.
- **Promotion:** Matches run in blocks of `block_games` (default `40`) with `opening_offset = 20 * block` and `seed = 1_000_000 * G + block`. After each block, compute the cumulative LLR (§9, `elo0 = 0`, `elo1 = 40`, α = β = 0.05). Promote when `LLR >= upper`. Reject when `LLR <= lower`. At `max_games` (default `400`), promote iff `LLR > 0`.
  - `decision.json`: `{"promoted":true,"reason":"sprt_accept|sprt_reject|max_games|bootstrap","games":120,"score":0.58,"elo":56.1,"elo_ci95":[12.3,100.6],"llr":2.97}`
- **Resign threshold update:** After a generation's self-play, compute the false-positive rate (`sum(false_positives) / sum(playthrough_games)`, skipped when there are 0 playthrough games). If it is `> 0.05`, set `resign_threshold = max(resign_threshold - 0.02, -0.99)` in `run.json`. Gen-1 starts at `-0.95`.
- **Elo ladder (`ladder.json`):** `[{"gen":0,"elo":0.0,"ci95":[0,0],"vs_random":0.97,"vs_material":0.81}, …]`. Each promoted gen `G` gets `elo = elo(best before G) + decision.elo`. `vs_*` are anchor-match scores of `anchors.games` (default `20`) games at the match `sims`.
- **Disk pruning (stage `prune`):**
  - Delete `selfplay/` of gens `< G-W+1`.
  - Delete `model.pt` and `model.onnx` of rejected candidates; keep their `train.json`, `match/` and `decision.json`.
  - If the run is still over `disk_budget_gb` (default `20`), delete `model.pt` of the oldest promoted gens (never the best gen). Their `.onnx` is kept for the ladder.
  - Never delete `supervised*/`, the best gen, or the current gen. If the run is still over budget, the loop exits non-zero with a message.
- **Config (`pipeline/configs/*.json`):**
~~~json
{ "seed": 0, "disk_budget_gb": 20,
  "selfplay": { "games": 1200, "games_per_chunk": 50, "sims": 200, "parallel": 32, "leaf_batch": 1,
                "max_plies": 512, "temp_plies": 30, "resign_consecutive": 3, "resign_playthrough": 0.1,
                "ep": "auto", "threads": 1 },
  "train":    { "steps": 4000, "batch_size": 512, "lr": 0.001, "weight_decay": 0.0001,
                "warmup_steps": 100, "steps_per_epoch": 500, "device": "auto" },
  "window":   { "generations": 4, "max_samples": 1000000, "supervised_weight": [0.5, 0.3, 0.2, 0.1, 0.0] },
  "match":    { "block_games": 40, "max_games": 400, "sims": 200, "batch": 8, "max_plies": 300,
                "elo0": 0, "elo1": 40, "alpha": 0.05, "beta": 0.05 },
  "anchors":  { "games": 20 } }
~~~

## 11. UCI Extensions
- **`go` limits:** `nodes N`, `movetime MS`, `wtime/btime/winc/binc MS`, `movestogo M`, `infinite`. When several are given, whichever limit is hit first stops the search. `infinite` searches until `stop` (still capped by the arena: once the arena is full, simulations continue without expanding).
- **Time budget** (side to move has `rem` ms and `inc` ms increment): `mtg = movestogo or 30`; `budget = rem/mtg + 0.75*inc`; `budget = min(budget, rem/4) - MoveOverhead`; floor `10 ms`. `movetime` overrides the clock formula.
- **Options** (printed after `id`, before `uciok`):
  - `option name ModelPath type string default <empty>` (empty = uniform net)
  - `option name Threads type spin default 1 min 1 max 8`
  - `option name BatchSize type spin default 8 min 1 max 64`
  - `option name MoveOverhead type spin default 50 min 0 max 5000`
- **Async search:** `go` returns at once. The search runs on a worker thread. `stop` ends it, and `bestmove` follows within 100 ms. `isready` is answered during search. `info nodes N nps X time T score … pv …` is printed at least every 500 ms and once at the end.
- **Repetition awareness:** The game history from `position … moves …` (position hashes) is passed to the search. A non-root node whose position already occurs in the history or earlier on its own path is a terminal draw (`value 0`).
- **Tree reuse:** When a new `position` command extends the previous one by 1–2 plies, the matching subtree becomes the new root (§7 compaction). Otherwise the tree is reset. `ucinewgame` always resets it.

## Open Questions (do not act on without the owner's decision)
- §4 requires the CoreML EP. §8 restricts CoreML to macOS so Linux gates can run. Confirm the `ort` `coreml` feature still compiles on Linux; if it does not, move it to a macOS-only target dependency.
- CLAUDE.md quotes the arena as "~4MB" while `arena.rs` measures ~1.8 MB (28 B/node). The cap (65,536) is what matters; the number in the text is out of date.
