"""Store round trip (PLAN §3.2): run_id, the flat run.json schema, the registry."""

import numpy as np
import pandas as pd
import pytest

from gsp import HARNESS_VERSION
from gsp.store import ids, index, paths, records
from gsp.store.io import load_npz, npz_bytes, save_npz


def base_config(**kw):
    cfg = dict(arm="A1", encoding="confined", inst_id="N05e003q1.5", K=12, rule="violation",
               connectivity="ring", effort_kind="depth", effort=5, restart=0, lam=None,
               schedule=None, seed=4001 + 4099 * 3 + 4999 * 5, seed_ga=6007 + 6101 * 3 + 6199 * 5)
    cfg.update(kw)
    return cfg


def test_ids():
    assert ids.draw_id(5, 3) == "N05e003"
    assert ids.inst_id(5, 3, 1.5) == "N05e003q1.5"
    assert ids.inst_id(10, 29, 1.0) == "N10e029q1.0"
    assert ids.inst_id(4, 112, 3) == "N04e112q3.0"
    assert ids.parse_inst_id("N05e003q1.5") == {"N": 5, "e": 3, "q": 1.5, "version": None}
    assert ids.parse_inst_id("N05e003q1.5v2")["version"] == "v2"
    with pytest.raises(ValueError):
        ids.parse_inst_id("N5e3q1.5")


def test_run_id_deterministic_and_sensitive():
    cfg = ids.with_version(base_config())
    assert cfg["harness_version"] == HARNESS_VERSION
    rid = ids.run_id(cfg)
    assert len(rid) == 16 and int(rid, 16) >= 0
    # key order and numpy scalar types do not matter
    shuffled = dict(reversed(list(cfg.items())))
    shuffled["K"] = np.int64(12)
    assert ids.run_id(shuffled) == rid
    # every config field matters, including the harness version
    for k, v in [("K", 24), ("restart", 1), ("lam", 0.005), ("effort", 7), ("inst_id", "N05e003q3.0"),
                 ("harness_version", "0.1.1"), ("extra_flag", True)]:
        assert ids.run_id({**cfg, k: v}) != rid, k
    with pytest.raises(KeyError):
        ids.run_id(base_config())               # no harness_version
    with pytest.raises(ValueError):
        ids.canonical_json({"x": float("nan")})


def test_git_sha_not_hashed():
    r1 = records.new_record(base_config())
    r2 = dict(r1, git_sha="deadbeef", host="elsewhere", started_at="2000-01-01T00:00:00+00:00")
    records.validate_record(r2)
    assert r1["run_id"] == r2["run_id"]


def test_record_roundtrip_and_registry(tmp_path):
    rec = records.new_record(base_config(), runtime={"cudaq_version": "0.15.1", "target": "nvidia",
                                                     "target_option": "fp64", "driver_version": "560.35.03",
                                                     "gpu_name": "RTX"})
    assert rec["status"] == "running"
    records.write_record(rec, root=tmp_path)
    assert not records.is_done("A1", rec["run_id"], root=tmp_path)
    records.mark_done(rec, metrics={"ar_f": 0.71, "p_feas": np.float64(1.0)}, timings={"train": 3.5},
                      diagnostics={"n_iter": 42}, wall_s=4.0)
    p = records.write_record(rec, root=tmp_path)
    assert p == paths.run_dir("A1", rec["run_id"], tmp_path) / "run.json"
    back = records.read_record(p)
    assert back == rec
    assert records.is_done("A1", rec["run_id"], root=tmp_path)
    # a failed run in another arm, and a second done run
    rec2 = records.new_record(base_config(arm="A0", encoding="penalty", K=None, rule=None,
                                          connectivity=None, lam=0.005, seed_ga=None))
    try:
        raise RuntimeError("boom")
    except RuntimeError as exc:
        records.mark_failed(rec2, exc)
    records.write_record(rec2, root=tmp_path)
    assert "boom" in records.read_record(records.run_json_path(rec2, root=tmp_path))["error"]
    df = index.build_index(tmp_path)
    assert paths.registry_path(tmp_path).exists()
    assert len(df) == 2 and set(df["status"]) == {"done", "failed"}
    reg = index.load_registry(tmp_path)
    row = reg[reg["run_id"] == rec["run_id"]].iloc[0]
    for k in ("arm", "inst_id", "K", "effort", "metric_ar_f", "time_train", "diag_n_iter", "cudaq_version"):
        assert row[k] == rec[k], k
    assert pd.isna(reg[reg["run_id"] == rec2["run_id"]].iloc[0]["metric_ar_f"])
    with pytest.raises(FileNotFoundError):
        index.load_metrics(tmp_path)


def test_record_validation():
    rec = records.new_record(base_config())
    bad = dict(rec, K=13)                     # config changed after hashing
    with pytest.raises(ValueError):
        records.validate_record(bad)
    with pytest.raises(TypeError):
        records.new_record(base_config(schedule=[0.2, 3.0]))    # not flat
    with pytest.raises(KeyError):
        records.new_record({k: v for k, v in base_config().items() if k != "rule"})
    with pytest.raises(KeyError):
        records.new_record(base_config(metric_x=1.0))           # reserved prefix


def test_npz_bytes_deterministic(tmp_path):
    arrays = {"a": np.arange(5), "s": np.array("N05e000q1.5"), "m": np.eye(3)}
    assert npz_bytes(arrays) == npz_bytes(dict(arrays))
    save_npz(tmp_path / "x.npz", arrays)
    back = load_npz(tmp_path / "x.npz")
    assert back.keys() == arrays.keys()
    assert all(np.array_equal(back[k], arrays[k]) and back[k].shape == arrays[k].shape for k in arrays)
    assert back["s"].ndim == 0 and str(back["s"]) == "N05e000q1.5"
    with pytest.raises(TypeError):
        npz_bytes({"o": np.array([{}], dtype=object)})
