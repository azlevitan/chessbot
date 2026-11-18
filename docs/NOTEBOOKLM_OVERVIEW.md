# How chessbot Works: A Beginner's Walkthrough

*Source document for NotebookLM. It is meant to be turned into a video walkthrough for a beginner computer scientist. It explains what the chessbot is, how each part works, why it was designed this way, and how it can be tuned and improved. It is written as plain prose with clear section headings so it can be narrated in order.*

---

## 0. Notes for the video producer (NotebookLM)

- **Audience:** a beginner computer scientist. They know basic programming (loops, arrays, functions) and have heard of neural networks. They have not studied reinforcement learning or chess engines.
- **Tone:** friendly and concrete. Use analogies first, then the real numbers.
- **Accuracy rule.** Use the current implementation and milestone status in Section 9. The CPU smoke run exercises the loop through a small generation. The full-scale M1 run and strength milestones are still open.
- **Suggested episode outline** (about 15–25 minutes):
  1. The big idea: an engine that learns chess by playing itself (Sections 1–2)
  2. How a computer "sees" a chessboard (Section 3)
  3. How it names every possible move with a number (Section 4)
  4. The neural network: intuition plus value (Section 5)
  5. Thinking ahead with Monte Carlo Tree Search (Section 6)
  6. Where training data comes from: human games and self-play (Section 7)
  7. Training in Python and handing the model back to Rust (Section 8)
  8. Project status and roadmap (Section 9)
  9. Fitting it all in 8 GB of RAM (Section 10)
  10. How to make it stronger: tuning and optimization (Section 11)
  11. Recap and glossary (Sections 12–13)

---

## 1. The big idea in one paragraph

chessbot is a chess engine and training pipeline inspired by DeepMind's **AlphaZero** and the open-source **Leela Chess Zero (Lc0)**. A **neural network** predicts promising moves and the probabilities of a win, draw, or loss. **Monte Carlo Tree Search (MCTS)** uses those predictions to guide its lookahead. The pipeline first trains on human games, then uses games the engine plays against itself to train candidates and compares each candidate with the current best network. Sustained improvement across generations is an open milestone.

**Analogy:** the neural network is a chess player's *intuition* ("this move feels right, and I think I'm slightly winning"). The tree search is the player's *calculation* ("if I go here, they go there, then I..."). Self-play is *practice*. Training is *reflecting on what happened*.

---

## 2. The two-language architecture

The project is split between two programming languages. Each one does the job it is best at.

### Rust: the `engine/` folder
Rust is a fast, memory-safe systems language. The Rust engine does everything that has to be fast or exact:
- It represents the board and generates legal moves, using the `cozy-chess` library.
- It **encodes** boards into numbers the neural network can read.
- It runs the neural network for inference, using **ONNX Runtime** (the `ort` library).
- It runs the MCTS search.
- It plays self-play games and writes training data to disk.
- It implements basic **UCI** (Universal Chess Interface), with synchronous node- and time-limited search.

### Python + PyTorch: the `training/` folder
PyTorch is the standard tool for building and training neural networks. The Python side:
- defines the neural network,
- reads the training data that Rust wrote,
- trains the network on Apple's GPU (the `mps` backend) or on a CPU,
- exports the trained network to an **ONNX** file that the Rust engine can load.

### The handshake between them
Two file formats carry the training records and weights across the language boundary:
1. **`.bin` training files**, written by Rust and read by Python.
2. **`.onnx` model files**, written by Python and read by Rust.

The `pipeline/` scripts run data generation, training, export, and evaluation as separate processes and track results in JSON sidecars and run metadata.

### The most important rule: one source of truth for encoding
**Rust owns board canonicalization and move indexing.** Python decodes the compact records into the input planes and targets used by PyTorch.

If the training and inference paths disagree about a detail such as which corner is square 0, the network sees different inputs when it plays. A shared encoding reduces that risk. The **contract test** (`training/tests/test_contract.py`) checks the Rust records and the Python decoder against an independent `python-chess` oracle.

---

## 3. How the computer "sees" a chessboard

A neural network cannot look at a picture of a board. It needs numbers. chessbot turns each position into a stack of **18 layers**, called **planes**. Each plane is an 8×8 grid of numbers, one number per square. Together they form an array of shape `18 × 8 × 8`.

| Planes | What they mean |
|---|---|
| 0–5 | Where **my** pieces are: pawns, knights, bishops, rooks, queens, king. A square is `1.0` if my piece of that type is there, otherwise `0.0`. |
| 6–11 | The same six planes for **the opponent's** pieces. |
| 12–15 | Castling rights: my kingside, my queenside, their kingside, their queenside. Each plane is all `1.0`s or all `0.0`s. |
| 16 | The en-passant target square, if there is one. |
| 17 | The 50-move-rule counter, divided by 100, filling the whole plane. |

**Analogy:** think of transparent sheets laid over the board. One sheet shows only my pawns, another only my knights, and so on. Stack all 18 sheets and you have the network's view of the position.

### Perspective normalization: always "my" point of view
The planes do not say "white" and "black". They say **"us"** (the side whose turn it is) and **"them"**. When it is Black's turn, the engine **flips the board vertically**. In code, every square index `sq` becomes `sq ^ 56`. That XOR flips the rank, so rank 1 swaps with rank 8. The engine also swaps the piece planes, so Black's pieces go into the "us" slots.

Why do this? The network only has to learn chess **once**, from one point of view. It never has to learn the same idea twice, once for White and once for Black. A useful bonus: flipping a whole 64-bit bitboard vertically is just a **byte swap** (`swap_bytes()`), which is one very fast CPU instruction.

### Bitboards
Inside the engine, each piece type is stored as a **bitboard**: a 64-bit integer with one bit per square. This is a classic chess-programming trick. It is compact (8 bytes per piece type) and very fast to work with.

---

## 4. Giving every move a number (the 4,168-move "action space")

The network's move output is a list of scores, one for every possible move. To make that work, every move needs a fixed number. chessbot uses **4,168** numbered slots:

- **Normal moves, including promotion to a queen:** `index = from_square × 64 + to_square`. That gives 64 × 64 = 4,096 slots, numbered 0–4095. Most of these are never legal (a pawn can't jump across the board), but a simple formula is easy to compute and easy to check.
- **Underpromotions** (promoting to a knight, bishop, or rook): 72 extra slots, numbered 4096–4167. There are 8 files, times 3 directions (capture left, straight, capture right), times 3 piece types.
- **Castling** is written as the king's two-square move (for example `e1g1`). That matches the UCI standard. The `cozy-chess` library stores castling internally as "king captures own rook". The conversion between the two forms happens in exactly one file, `engine/src/encoder/move_mapper.rs`.

Moves are also numbered from the side to move's perspective, so the flip from Section 3 applies to them too.

**Legal-move masking.** Search applies softmax to legal moves only. The training loss sets logits outside the stored policy entries to `-1e9` before softmax. The record holds up to 64 legal moves, as described in Section 7.

A test plays 1,000 random positions and checks that every legal move maps to a unique number and back again with no mistakes. In math terms, the mapping is a **bijection**.

---

## 5. The neural network: one brain, two heads

The network is called an **SE-ResNet-6x64**. It is defined in `training/models/net.py`.

### The body ("trunk")
- A **stem** convolution turns the 18 input planes into 64 **channels**, which you can think of as 64 learned feature maps.
- Then come **6 residual blocks**. Each block has two 3×3 convolutions with **batch normalization** and **ReLU**. Each block also adds its input back to its output. That is the "skip connection" from **ResNet**, and it makes deeper networks much easier to train.
- Each block also has a **Squeeze-and-Excitation (SE)** unit. It averages each channel over the whole board, passes those 64 numbers through a tiny two-layer network, and uses the result to turn each channel up or down. In plain words, the network can decide globally which features matter in this position, such as "king safety matters a lot here".

"6x64" means 6 blocks with 64 channels each. That is small on purpose so it can run on a laptop. For comparison, top Lc0 networks use 20 to 40 or more blocks and hundreds of channels.

### Two heads
The trunk splits into two outputs:
1. **Policy head:** 4,168 numbers (logits), one per move slot. After masking and softmax, they form a probability distribution: "how promising is each move?"
2. **Value head:** 3 numbers, for **Loss, Draw, and Win** (WDL), from the side to move's point of view. After softmax, the engine computes one score: `v = P(win) − P(loss)`. It ranges from −1 (certain loss) to +1 (certain win).

Predicting win, draw, and loss separately, rather than one score, tells the engine the difference between "dead equal and drawish" and "wild position where anything can happen".

### A useful fact for optimization
The network has about **9 million parameters** (about 36 MB in 32-bit floats). Roughly **95% of them sit in one layer**: the final fully connected layer of the policy head, which maps 32 × 64 = 2,048 features to 4,168 moves. The whole residual trunk is only about half a million parameters. Section 11 explains why this is one of the best places to optimize.

### How it learns: the loss function
Training minimizes two errors added together (`training/losses.py`):
- **Policy loss:** cross-entropy between the network's move probabilities (illegal moves masked) and the **target** probabilities. In self-play, the target is how often the search visited each move. For human games, the target is 1.0 on the move the human played.
- **Value loss.** Cross-entropy on the final game outcome for bootstrap and validation. With the default self-play loop settings, candidate training blends the game outcome with a soft target derived from the search's `q_value` at weight 0.3.

### Exporting to ONNX
After training, `training/export_onnx.py` saves a dynamic-batch **ONNX** file using opset 17. Export checks the tensor contract and compares both heads against CPU PyTorch. Seeded stress inputs use a per-logit bound of `1e-4 + 1e-6 * abs(reference)`. Bootstrap and candidate export also check held-out Rust-encoded positions with a raw-logit bound of `1e-4`. Both paths check softmax probabilities within `1e-5`. The Rust engine can request **CoreML** on macOS and uses CPU inference on Linux. [Export contract](../SPEC.md#4-neural-network--onnx-contract)

---

## 6. Thinking ahead: Monte Carlo Tree Search (MCTS)

The network gives quick intuition. **MCTS** turns that intuition into careful calculation. The code is in `engine/src/mcts/`.

### The search tree
The search grows a tree. Each **node** is a position, and each **edge** is a move. Every node stores:
- `prior` (P): how good the network thought the move into this node was,
- `visits` (N): how many times the search has passed through this node,
- `value_sum`: the total of all evaluations backed up through it. `Q = value_sum / visits` is the average.

### One simulation, in four steps
The search repeats one **simulation** hundreds or thousands of times, for example 800 by default in UCI mode or 200 per move in self-play.

1. **Select.** Start at the root. At each node, pick the child with the highest **PUCT** score:

   `score = Q + c_puct × P × √(N_parent) / (1 + N_child)`

   - `Q` is **exploitation**: "this move has worked well so far."
   - The second term is **exploration**: "the network likes this move, and we haven't tried it much yet." It shrinks as a move gets more visits.
   - `c_puct = 1.5` sets the balance between the two. An unvisited child starts at `Q = 0` at the root. Elsewhere, its initial value is the parent's value from the current side's perspective, reduced by 0.25 and clamped to `[-1, 1]`.
2. **Expand and evaluate.** When the search reaches a node it has not explored yet, it asks the neural network for move priors and a value. It then creates child nodes for every legal move. If the position is checkmate or a draw, the search uses the exact result instead.
3. **Back up.** Send the value back up the path to the root, **flipping its sign at each level**. A position that is good for me is bad for my opponent, so `+0.6` for one side is `−0.6` for the other.
4. **Repeat.**

After all simulations, the engine plays the **most-visited** move. Visit counts are steadier than raw averages, because a move only gets many visits if it kept looking good.

**Analogy:** imagine planning a road trip with a friend who has good instincts (the network). You mostly explore the routes your friend suggests, but you check out a new road now and then. Every time you scout a route, you note how it went. In the end you take the route you scouted most, because it kept holding up.

### A fixed-size memory "arena"
The tree does not allocate memory piece by piece. It lives in an **arena**: one pre-sized array that holds at most **65,536 nodes**. Each node is about 28 bytes, so the whole tree takes about 1.8 MB. A node's children sit next to each other in the array, so a node only stores "first child index" and "number of children". This approach is fast, friendly to the CPU cache, and has a hard memory limit. That matters on an 8 GB laptop (Section 10). If the arena fills up, the search keeps running but stops adding new nodes.

### Exploration noise in self-play
In self-play, the engine adds random **Dirichlet noise** to the move priors at the root (`alpha = 0.3`, `eps = 0.25`, the AlphaZero values). This makes it sometimes try moves the network underrates, so it can discover that they are actually good. Without noise, the engine would keep repeating its own habits and stop learning new ideas.

---

## 7. Where the training data comes from

Every training example is one position plus its answers. It is stored as a fixed-size **492-byte** record. The layout is defined in `SPEC.md` §3 and `engine/src/encoder/sample.rs`:
- the 12 piece bitboards (96 bytes),
- castling, en-passant and 50-move counter bytes,
- the game's final result from this side's view (Loss, Draw, or Win),
- the search's own evaluation of the position (`q_value`),
- up to 64 `(move number, target probability)` pairs, one for each legal move.

Storing the moves as a short list saves space. For positions with more than 64 legal moves, the encoder keeps the 64 highest-probability entries, breaks ties by move index, and renormalizes the retained mass.

### Source 1: human games (supervised "bootstrap")
`engine convert-pgn` reads standard-chess **PGN** games and writes a record for every mainline position before a move. The target is the move the human actually played, and the result label comes from the game outcome. The recorded Generation 0 run used about 4.2 million supervised positions. [Generation 0 results](../training/GENERATION_0_RESULTS.json)

### Source 2: self-play (reinforcement learning)
`engine selfplay` makes the engine play itself:
- At every move it runs MCTS (200 simulations by default) with root noise.
- For the first **30 plies** (half-moves), it picks moves **at random, weighted by visit counts**. This is called "temperature 1". It creates variety in the openings. After ply 30, it plays the most-visited move.
- A game ends by checkmate, stalemate, the 50-move rule, threefold repetition, a basic insufficient-material check, resignation, or the 512-ply limit. At that limit self-play labels the game a draw. A play-through sample of games continues after a would-be resignation to audit the threshold.
- When the game ends, each saved position gets the final result **from that position's side to move**.

The policy target for self-play is the search's **visit distribution**. This is the key idea of AlphaZero. The search, with its look-ahead, is smarter than the raw network, so training the network to predict what the search would do **pulls the network's intuition up toward the search's calculation**.

---

## 8. Training in Python

`training/data/chunk_dataset.py` indexes multiple read-only **`numpy.memmap`** files. It gathers the requested records and uses the decoder in `training/data/replay_buffer.py` to expand bitboards into planes and sparse moves into dense training targets. Resident memory depends on the working set, decoded batches, and model.

The training loop computes the policy and value losses and updates the weights with **AdamW**, using linear warm-up followed by cosine decay. The trained model is exported to ONNX for Rust inference.

### Tests that prove the plumbing works
- **Contract gate** (`test_contract.py`): Rust writes test positions, Python reads them, and every plane, flag and move index is checked against `python-chess`.
- **Overfit gate** (`test_overfit.py`): the network must be able to memorize one small batch, with loss below 0.05. If a network can't even memorize 16 examples, something is wired wrong.
- **PGN contract** (`test_pgn_contract.py`): checks the PGN-to-binary conversion.
- **Perft test.** Checks the known count of **197,281** move sequences at depth four from the starting position.
- **Mate-in-one UCI test** (`pipeline/test_uci_mate.py`): starts the real engine binary and checks that it finds checkmate in three positions.

---

## 9. Project status

The project is built in **phases**. Each phase ends with a **gate**: a test command that must pass before the phase counts as done.

The current branch includes:
1. Rust board wrapper, move numbering, binary encoder (GATE 1: perft and move round-trips).
2. Python memmap reader and the cross-language contract test (GATE 2).
3. The SE-ResNet network, the masked losses, ONNX export, and the overfit test (GATE 3).
4. ONNX inference in Rust, arena MCTS, and a basic UCI loop (GATE 4: finds mate-in-one).
5. PGN conversion and self-play data generation with Dirichlet noise (GATE 5).

Later phases add chunked training data, the streaming trainer and supervised Generation 0 (Phase 6), batched search, subtree reuse, parallel self-play and audited resignation (Phase 7), paired matches and promotion statistics (Phase 8), and a resumable generation loop with pruning and reporting (Phase 9). Phase 10 tracks game-clock management, `stop`, live `info`, runtime options, history-aware UCI search, and optional Stockfish calibration. [Roadmap](../TASKS.md)

### How a generation runs

1. The current best network generates self-play data. The overnight preset requests 1,200 games.
2. The trainer starts a candidate from the best checkpoint. The default replay window selects whole chunks from up to four generations, stopping when it reaches or crosses one million self-play positions. Supervised sampling weight starts at 50% and reaches 0% at generation 5.
3. Export validates the candidate in ONNX Runtime. The candidate and incumbent play paired openings with colors swapped, in blocks of 40 games by default.
4. A GSPRT approximation uses the running results to compare the 0 and +40 relative Elo hypotheses. At the game cap, the sign of the statistic breaks an undecided comparison. The opening book wraps after 50 pairs, and the calculation treats game results individually. [Evaluation specification](../SPEC.md#9-matches--evaluation)
5. The loop runs anchor matches for promoted candidates, prunes eligible artifacts, and writes `ladder.json` and `report.tsv`. The ladder is relative to Generation 0.

Small CPU tests exercise the implemented loop, including crash and resume. The supervised Generation 0 milestone is owner-reported. Full overnight timing, stage memory usage, and sustained strength gains on the target hardware are open milestones. [Generation 0 results](../training/GENERATION_0_RESULTS.json)

---

## 10. Engineering for a small laptop (8 GB Apple M1)

The project is designed to train on an Apple M1 MacBook Pro with only **8 GB of RAM** shared between the CPU and GPU. That shapes many decisions:
- **Never load the whole dataset into memory.** Always memmap it and decode lazily.
- **`num_workers=0` and `pin_memory=False`** in PyTorch's DataLoader. On macOS, worker processes can duplicate memory and crash the machine.
- **Cap every search tree at 65,536 nodes** (about 1.8 MB).
- **Call `torch.mps.empty_cache()` after every epoch** to release GPU memory.
- **Run heavy pipeline stages as separate processes in sequence** to avoid keeping the trainer and inference workloads resident together. The overnight acceptance target is at most 3.5 GB peak RSS for each stage.
- **Keep every source file under 250 lines**, so each piece is easy to read, review and test.

Linux CPU tests exercise the cross-language integration without Apple hardware. The [README](../README.md#verification) lists the commands and current fixture issue.

---

## 11. How it can be fine-tuned and optimized

This is the part to experiment with. The ideas are grouped from easiest to hardest. One golden rule applies throughout: **change one thing at a time, and measure it with a match**. Phase 8's SPRT and Elo tools exist so you can tell real progress from luck.

### A. Search settings (no retraining needed)
- **`c_puct`** (currently 1.5): the exploration vs. exploitation balance. Higher values explore more. Lc0 scales it with visit count; trying values from 1.0 to 3.0 is a classic first experiment.
- **First-Play Urgency (FPU).** The implemented root and interior-node defaults are described in Section 6. Their effect on playing strength can be tested separately.
- **Simulations per move:** more simulations mean stronger play but slower games. In self-play, you can trade game count for game quality.
- **Batched evaluation with virtual loss.** A call can evaluate multiple pending positions. The batch can be smaller than the requested limit when paths collide or reach terminal positions. Throughput depends on model, provider and batch size. The recorded M1 single-tree comparison favored batch 1 among the tested sizes. [Benchmark results](../TASKS.md#phase-7-search-throughput-repetition--fpu-batched-mcts-tree-reuse-parallel-self-play)
- **Tree reuse** (Phase 7): keep the part of the tree under the move that was actually played, so work isn't thrown away between moves.

### B. Self-play and data settings
- **Dirichlet noise** (`alpha = 0.3`, `eps = 0.25`): more noise means more exploration but noisier training targets.
- **Temperature plies** (30): controls how much variety the openings get.
- **Resignation threshold.** The loop starts at −0.95 and plays out 10% of games by default to estimate how often a would-be resignation gives up a draw or win.
- **Replay window** (4 generations, 1M samples) and the **human-data weights** (`[0.5, 0.3, 0.2, 0.1, 0.0]`): how much old data to keep mixing in, and how fast to wean the engine off human games.
- **Value target blending.** `train.q_ratio` controls the contribution of the stored `q_value` to candidate training. The default is 0.3.

### C. Training settings
- **Learning rate, warm-up, and schedule** (defaults: 1e-3, AdamW, up to 100 warm-up steps, cosine decay to 10%).
- **Batch size** (512) and **steps per generation** (4,000).
- **Weight decay** (1e-4) to discourage overfitting.
- **Loss weighting:** scale the value loss relative to the policy loss. Value heads overfit easily on self-play data.
- **Data augmentation:** chess is not symmetric left-to-right once castling rights exist, so this is limited. A mirrored board is valid only when neither side can castle.

### D. Network architecture
- **Shrink the policy head.** About 95% of the parameters are in one fully connected layer (2,048 → 4,168). Lc0 uses a **convolutional policy head** that outputs move scores per square, which uses far fewer parameters. This would make the model smaller and faster. However, it changes the ONNX contract, so `SPEC.md` would need updating first.
- **Deeper or wider trunk:** 10 blocks × 128 channels would be much stronger per evaluation but slower. On a laptop the right size is a speed-vs-quality trade-off you have to measure.
- **More input history:** AlphaZero and Lc0 feed the last 8 positions so the network can "see" repetitions and recent moves. Currently there is 1. This would change the 18-plane contract.
- **Other heads:** extra outputs such as "moves left in the game" (used by Lc0) can improve endgame play and time management.

### E. Inference speed
- **CoreML and CPU.** The overnight preset requests CoreML. Benchmark results label the requested provider, which can fall back to CPU. [Provider measurements](../TASKS.md#phase-7-search-throughput-repetition--fpu-batched-mcts-tree-reuse-parallel-self-play)
- **Lower precision:** converting the model to float16 or int8 can roughly double throughput, with a small accuracy cost.
- **Threads** for ONNX Runtime, and how many games run in parallel.

### F. Measuring progress
- **Relative Elo** between generations, using paired matches and the implemented GSPRT approximation. Reported intervals describe the individual comparisons and depend on the match design.
- **Anchor matches** against a random mover and a material counter, as sanity checks.
- **Absolute Elo** by playing Stockfish at fixed strength levels (planned Phase 10).
- **Training metrics:** validation policy top-1 accuracy (the Milestone 1 target is ≥ 35% on held-out human games) and value loss.

---

## 12. Recap

1. Rust turns a chessboard into 18 planes from the side to move's point of view, including the normalized halfmove clock.
2. A neural network reads those layers and outputs move probabilities plus a win/draw/loss estimate.
3. Monte Carlo Tree Search uses the network to look ahead and picks the most-visited move.
4. Self-play games record what the search decided and who won. Rust saves them as compact 492-byte records.
5. Python trains the network to predict the search's choices and the game results, then exports it to ONNX for Rust to use.
6. A resumable loop scores each candidate against the incumbent and records the promotion decision.
7. Contract tests check compatibility across the two languages. Memory-mapped data, bounded arenas, and sequential heavy stages limit memory use.

---

## 13. Glossary

- **AlphaZero / Leela Chess Zero (Lc0):** famous engines that learn chess through self-play using a neural network plus MCTS. chessbot follows their design.
- **Bitboard:** a 64-bit integer with one bit per square, used to store where pieces are.
- **Bijection:** a one-to-one mapping in both directions. Every move has exactly one number, and every number has at most one move.
- **Cross-entropy:** a standard loss that measures how far predicted probabilities are from target probabilities.
- **CoreML:** Apple's on-device machine-learning accelerator framework.
- **Dirichlet noise:** a kind of random noise that produces a random probability distribution, used to encourage exploration.
- **Elo:** a rating system. A 100-point difference means the stronger side scores about 64%.
- **Logit:** a raw network output score before it is turned into a probability by softmax.
- **memmap (memory map):** a way to treat a file on disk as an in-memory array without loading all of it.
- **MCTS (Monte Carlo Tree Search):** a search algorithm that repeatedly samples promising lines and averages their results.
- **ONNX / ONNX Runtime:** a portable neural-network file format, and the library that runs it.
- **PGN:** Portable Game Notation, the standard text format for recorded chess games.
- **Perft:** "performance test". It counts all legal move sequences to a fixed depth to verify a move generator.
- **Ply:** one move by one side (a half-move).
- **Policy / Value:** the network's two outputs. Policy is "which moves look good", value is "who's winning".
- **PUCT:** the formula MCTS uses to pick which move to explore next, balancing the network's prior against observed results.
- **Residual block / skip connection:** a network layer that adds its input back to its output, making deep networks trainable.
- **Softmax:** turns a list of scores into probabilities that sum to 1.
- **SPRT:** Sequential Probability Ratio Test, a statistical method that decides "better or not" using as few games as possible.
- **Squeeze-and-Excitation (SE):** a small add-on that lets a network re-weight its feature channels based on the whole board.
- **UCI:** Universal Chess Interface, the text protocol chess GUIs use to talk to engines.
- **Virtual loss:** a trick for batched MCTS. It temporarily marks paths as "being explored" so parallel simulations spread out instead of piling onto the same line.
- **WDL:** Win / Draw / Loss, the three-way value output. Tensor order is `[Loss, Draw, Win]`.

---

## 14. Where to look in the code

| Topic | File |
|---|---|
| All data contracts | `SPEC.md` |
| Roadmap and gates | `TASKS.md` |
| Board → planes, samples | `engine/src/encoder/tensor_encoder.rs` |
| Move ↔ number mapping | `engine/src/encoder/move_mapper.rs` |
| 492-byte record layout | `engine/src/encoder/sample.rs` |
| MCTS search and PUCT | `engine/src/mcts/search.rs`, `engine/src/mcts/tree.rs` |
| Fixed-size node arena | `engine/src/mcts/arena.rs` |
| Dirichlet noise | `engine/src/mcts/noise.rs` |
| Neural-net inference in Rust | `engine/src/nn/ort_infer.rs`, `engine/src/nn/mod.rs` |
| Self-play games | `engine/src/selfplay/game.rs` |
| PGN conversion | `engine/src/cli/convert_pgn.rs` |
| UCI protocol | `engine/src/uci/uci_loop.rs` |
| Network definition | `training/models/net.py` |
| Losses | `training/losses.py` |
| Data reader | `training/data/replay_buffer.py` |
| ONNX export | `training/export_onnx.py` |
| Resumable generation loop and reports | `pipeline/loop.py`, `pipeline/stages.py`, `pipeline/report.py` |
