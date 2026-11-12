"""Config and replay-window contracts using metadata and fake Rust-sized chunks."""

import json
from pathlib import Path

import pytest

from config import load_config
from run_layout import REPO_ROOT, generation_dir
from window import select_window, train_args


def config_file(root, value):
    path = root / "config.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


@pytest.fixture
def cfg(tmp_path):
    return load_config(config_file(tmp_path, {}))


def chunk(directory, index=0, samples=2):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"chunk_{index:06d}.bin"
    path.write_bytes(bytes(samples * 492))
    path.with_suffix(".json").write_text(json.dumps(
        {"format": "chessbot-chunk-v1", "samples": samples}), encoding="utf-8")
    return path


def selfplay(root, gen, index=0, samples=2):
    return chunk(generation_dir(root, gen) / "selfplay", index, samples)


def test_newest_first_exact_generation_window_and_exclusions(tmp_path, cfg):
    paths = {g: [selfplay(tmp_path, g, i) for i in range(2)] for g in range(8)}
    chunk(tmp_path / "supervised")
    chunk(tmp_path / "supervised_val")
    expected = [p for g in (6, 5, 4, 3) for p in reversed(paths[g])]
    assert select_window(tmp_path, 6, cfg) == [(expected, 1.0)]
    assert select_window(tmp_path, 6, cfg) == [(expected, 1.0)]


@pytest.mark.parametrize("cap,count", [(1, 1), (3, 1), (4, 2), (7, 2), (8, 3)])
def test_whole_chunk_cap_equality_and_overshoot(tmp_path, cfg, cap, count):
    newest = selfplay(tmp_path, 3, 1, 3)
    older = selfplay(tmp_path, 3, 0, 4)
    previous = selfplay(tmp_path, 2, 0, 2)
    supervised = chunk(tmp_path / "supervised", samples=20)
    cfg["window"]["max_samples"] = cap
    assert select_window(tmp_path, 3, cfg) == [
        ([newest, older, previous][:count], 0.8), ([supervised], 0.2)]


@pytest.mark.parametrize("gen,weight", [(1, .5), (2, .3), (3, .2), (4, .1),
                                               (5, 0), (6, 0), (30, 0)])
def test_supervised_schedule_and_pooled_sources(tmp_path, cfg, gen, weight):
    replay = selfplay(tmp_path, gen)
    second = chunk(tmp_path / "supervised", 1)
    first = chunk(tmp_path / "supervised", 0)
    chunk(tmp_path / "supervised_val")
    expected = [([replay], 1 - weight)]
    if weight:
        expected.append(([first, second], weight))
    assert select_window(tmp_path, gen, cfg) == expected


def test_weight_one_missing_and_empty_sources(tmp_path, cfg):
    replay = selfplay(tmp_path, 1)
    assert select_window(tmp_path, 1, cfg) == [([replay], .5)]
    supervised = chunk(tmp_path / "supervised")
    cfg["window"]["supervised_weight"] = [1.0]
    assert select_window(tmp_path, 1, cfg) == [([supervised], 1.0)]
    assert select_window(tmp_path, 20, cfg) == [([supervised], 1.0)]
    supervised.unlink()
    with pytest.raises(ValueError):
        select_window(tmp_path, 1, cfg)
    cfg["window"]["supervised_weight"] = [0.0]
    with pytest.raises(ValueError):
        select_window(tmp_path, 20, cfg)


@pytest.mark.parametrize("fault", ["missing_json", "bad_json", "array", "format",
    "bool", "negative", "float", "size", "missing_bin", "bin_tmp", "json_tmp", "empty", "utf8"])
def test_incomplete_and_invalid_chunks_are_ignored(tmp_path, cfg, fault):
    valid = selfplay(tmp_path, 1)
    invalid = selfplay(tmp_path, 1, 1)
    sidecar = invalid.with_suffix(".json")
    metadata = {"format": "chessbot-chunk-v1", "samples": 2}
    if fault == "missing_json":
        sidecar.unlink()
    elif fault == "bad_json":
        sidecar.write_text("{", encoding="utf-8")
    elif fault == "utf8":
        sidecar.write_bytes(b"\xff")
    elif fault == "array":
        sidecar.write_text("[]", encoding="utf-8")
    elif fault in {"format", "bool", "negative", "float"}:
        field, value = ("format", "other") if fault == "format" else (
            "samples", {"bool": True, "negative": -1, "float": 2.0}[fault])
        metadata[field] = value
        sidecar.write_text(json.dumps(metadata), encoding="utf-8")
    elif fault == "size":
        invalid.write_bytes(b"short")
    elif fault == "missing_bin":
        invalid.unlink()
    elif fault == "bin_tmp":
        invalid.rename(invalid.with_name(invalid.name + ".tmp"))
    elif fault == "json_tmp":
        sidecar.rename(sidecar.with_name(sidecar.name + ".tmp"))
    else:
        chunk(invalid.parent, 1, 0)
    assert select_window(tmp_path, 1, cfg) == [([valid], .5)]


def test_selection_reads_metadata_only_and_keeps_noncanonical_files(tmp_path, cfg, monkeypatch):
    valid = selfplay(tmp_path, 1)
    stray = valid.with_name("other.bin")
    stray.write_bytes(valid.read_bytes())
    stray.with_suffix(".json").write_text(valid.with_suffix(".json").read_text())
    original_open = Path.open

    def metadata_only(path, *args, **kwargs):
        assert path.suffix != ".bin", "selection must not open binary sample data"
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", metadata_only)
    assert select_window(tmp_path, 1, cfg) == [([valid], .5)]
    assert stray.is_file()


@pytest.mark.parametrize("gen", [0, -1, True, 1.5, "1"])
def test_invalid_generation(tmp_path, cfg, gen):
    with pytest.raises(ValueError):
        select_window(tmp_path, gen, cfg)


def test_train_arguments_preserve_source_groups_paths_and_weights(tmp_path, cfg):
    run = tmp_path / "run with spaces"
    replay = selfplay(run, 1)
    first, second = [chunk(run / "supervised", i) for i in range(2)]
    assert train_args(select_window(run, 1, cfg)) == [
        "--source", f"{replay}:0.5", "--source", f"{first},{second}:0.5"]
    with pytest.raises(ValueError):
        train_args([([Path("ambiguous,path.bin")], 1.0)])


def test_train_arguments_omit_inactive_sources():
    path = Path("relative:source/run/chunk_000000.bin")
    assert train_args([([], .5), ([path], 0), ([path], 1)]) == ["--source", f"{path}:1.0"]


@pytest.mark.parametrize("sources", [[], [([], 1)], [([Path("a.bin")], 0)],
    [([Path("a.bin")], -1)], [([Path("a.bin")], True)], [([Path("a.bin")], 10**400)],
    [([Path("a.bin")], float("nan"))], [([Path("a.bin")], float("inf"))]])
def test_invalid_or_unusable_train_arguments(sources):
    with pytest.raises(ValueError):
        train_args(sources)


def test_config_deep_merge_list_replacement_and_independence(tmp_path, cfg):
    path = config_file(tmp_path, {"train": {"lr": .004},
        "window": {"supervised_weight": [1, 0]}, "selfplay": {"temp_plies": 0}})
    loaded = load_config(path)
    assert loaded["train"] == {**cfg["train"], "lr": .004}
    assert loaded["window"]["supervised_weight"] == [1, 0]
    assert loaded["selfplay"] == {**cfg["selfplay"], "temp_plies": 0}
    loaded["window"]["supervised_weight"].append(.7)
    loaded["train"]["batch_size"] = 1
    assert load_config(path)["window"]["supervised_weight"] == [1, 0]
    assert load_config(config_file(tmp_path, {})) == cfg
    assert cfg["train"]["q_ratio"] == .3 and cfg["selfplay"]["temp_plies"] == 30


def test_presets_match_required_values(cfg):
    directory = REPO_ROOT / "pipeline" / "configs"
    overnight = load_config(directory / "m1_overnight.json")
    assert overnight == {**cfg, "selfplay": {**cfg["selfplay"], "ep": "coreml"}}
    tiny = load_config(directory / "ci_tiny.json")
    overrides = {"selfplay": {"games": 4, "games_per_chunk": 2, "sims": 16,
        "parallel": 4, "max_plies": 40, "temp_plies": 10, "ep": "cpu"},
        "train": {"steps": 20, "batch_size": 16, "q_ratio": .3,
                  "device": "cpu", "steps_per_epoch": 10},
        "match": {"block_games": 4, "max_games": 8, "sims": 16, "batch": 1, "max_plies": 40},
        "anchors": {"games": 2}, "window": {"max_samples": 10000}}
    assert tiny == {**cfg, **{key: {**cfg[key], **value} for key, value in overrides.items()}}


@pytest.mark.parametrize("value", [[], None, 3, "text", {"train": []}, {"selfplay": None}])
def test_invalid_config_structure(tmp_path, value):
    with pytest.raises(ValueError):
        load_config(config_file(tmp_path, value))


def test_malformed_config_json(tmp_path):
    path = config_file(tmp_path, {})
    path.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(path)


@pytest.mark.parametrize("field,value", [
    ("seed", True), ("seed", -1), ("disk_budget_gb", 0), ("disk_budget_gb", float("inf")),
    ("selfplay.games", 0), ("selfplay.games_per_chunk", 1001), ("selfplay.sims", 1),
    ("selfplay.parallel", 1.5), ("selfplay.leaf_batch", False), ("selfplay.max_plies", -1),
    ("selfplay.temp_plies", -1), ("selfplay.threads", 0), ("selfplay.ep", "gpu"),
    ("selfplay.resign_consecutive", 0), ("selfplay.resign_playthrough", 1.1),
    ("train.steps", 0), ("train.batch_size", True), ("train.lr", 0),
    ("train.lr", float("nan")), ("train.weight_decay", -1), ("train.warmup_steps", -1),
    ("train.steps_per_epoch", 0), ("train.q_ratio", -0.1), ("train.device", "cuda"),
    ("window.generations", 0), ("window.max_samples", 0), ("window.supervised_weight", []),
    ("window.supervised_weight", [True]), ("window.supervised_weight", [float("nan")]),
    ("window.supervised_weight", [1.1]), ("match.block_games", 3), ("match.max_games", 5),
    ("match.sims", 1), ("match.batch", 0), ("match.max_plies", 0),
    ("match.elo1", 0), ("match.alpha", 0), ("match.beta", 1), ("anchors.games", 3)])
def test_invalid_config_values(tmp_path, field, value):
    parts = field.split(".")
    override = {parts[0]: value if len(parts) == 1 else {parts[1]: value}}
    with pytest.raises(ValueError):
        load_config(config_file(tmp_path, override))


def test_partial_final_chunks_and_match_blocks_are_valid(tmp_path):
    value = {"selfplay": {"games": 5, "games_per_chunk": 2},
             "match": {"block_games": 6, "max_games": 10},
             "train": {"steps": 20, "warmup_steps": 100}}
    assert load_config(config_file(tmp_path, value))["match"]["max_games"] == 10
