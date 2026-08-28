"""
微电网氢能交易系统 - 合成数据生成器
每次 reset() 调用生成一组带随机扰动的日内功率曲线。
步数和步长由 config 中 episode_length / dt 决定。
"""

import csv
import json
from pathlib import Path
import numpy as np


_ITALIAN_CACHE = {}
_ITALIAN_SPLIT_CACHE = {}
_ITALIAN_FIXED_SPLIT_CACHE = {}


def _gauss(h, mu, sigma):
    """标准高斯核（未归一化），用于构造负荷曲线形状。"""
    return np.exp(-0.5 * ((h - mu) / sigma) ** 2)


def _load_italian_data(path):
    path = str(Path(path).expanduser())
    if path in _ITALIAN_CACHE:
        return _ITALIAN_CACHE[path]
    rows = []
    header = None
    with open(path, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            first = row[0].strip()
            if header is None:
                if first.startswith("Ppv"):
                    header = [x.strip() for x in row if x.strip()]
                continue
            values = row[: len(header)]
            if len(values) < len(header):
                continue
            rows.append([float(x) for x in values])
    if header is None or not rows:
        raise ValueError(f"Invalid Italian profile file: {path}")
    data = np.asarray(rows, dtype=np.float32)
    groups = {
        "pv": [i for i, c in enumerate(header) if c.startswith("Ppv")],
        "wt": [i for i, c in enumerate(header) if c.startswith("Pw")],
        "load_e": [i for i, c in enumerate(header) if c.startswith("PL")],
    }
    _ITALIAN_CACHE[path] = (header, data, groups)
    return _ITALIAN_CACHE[path]


def _load_italian_split_manifest(path):
    path = str(Path(path).expanduser())
    if path in _ITALIAN_SPLIT_CACHE:
        return _ITALIAN_SPLIT_CACHE[path]
    with open(path) as f:
        manifest = json.load(f)
    _ITALIAN_SPLIT_CACHE[path] = manifest
    return manifest


def _split_columns(indices, n_groups):
    return [list(x) for x in np.array_split(np.asarray(indices, dtype=np.int64), n_groups)]


def _scale_group(data, day_slice, columns, cap):
    if cap <= 0.0 or not columns:
        return np.zeros(day_slice.stop - day_slice.start, dtype=np.float32)
    full = np.sum(data[:, columns], axis=1)
    denom = float(np.max(full))
    if denom <= 0.0:
        return np.zeros(day_slice.stop - day_slice.start, dtype=np.float32)
    profile = np.sum(data[day_slice, :][:, columns], axis=1)
    return np.clip(profile / denom * cap, 0.0, cap).astype(np.float32)


def _normalize_day_indices(day_indices, n_days):
    days = [int(day) for day in day_indices]
    if not days:
        raise ValueError("Italian day index pool is empty")
    invalid = [day for day in days if day < 0 or day >= n_days]
    if invalid:
        raise ValueError(f"Italian day indices out of range 0..{n_days - 1}: {invalid}")
    return sorted(set(days))


def _build_fixed_random_splits(n_days, train_ratio=0.70, seed=42):
    train_ratio = float(train_ratio)
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"italian_split_train_ratio must be in (0, 1), got {train_ratio}")
    if n_days < 2:
        raise ValueError("Italian fixed split requires at least two complete days")

    key = (int(n_days), float(train_ratio), int(seed))
    if key not in _ITALIAN_FIXED_SPLIT_CACHE:
        rng = np.random.RandomState(int(seed))
        permuted = [int(day) for day in rng.permutation(int(n_days))]
        train_count = int(round(int(n_days) * train_ratio))
        train_count = min(max(train_count, 1), int(n_days) - 1)
        train_days = sorted(permuted[:train_count])
        test_days = sorted(permuted[train_count:])
        _ITALIAN_FIXED_SPLIT_CACHE[key] = {
            "all": list(range(int(n_days))),
            "train": train_days,
            "test": test_days,
        }
    return _ITALIAN_FIXED_SPLIT_CACHE[key]


def _select_italian_day(config, rng, n_days, window_days=1):
    manual_days = config.get("italian_day_indices")
    if manual_days is not None:
        days = _normalize_day_indices(manual_days, n_days)
        split_name = "manual"
    elif config.get("italian_split_enable", False):
        split_name = str(config.get("italian_split_name", "train"))
        if split_name == "all":
            days = list(range(n_days))
        else:
            split_strategy = str(config.get("italian_split_strategy", "fixed_random"))
            if split_strategy == "manifest":
                manifest_path = config.get("italian_split_manifest_path")
                if not manifest_path:
                    raise ValueError("italian_split_manifest_path is required when italian_split_strategy='manifest'")
                manifest = _load_italian_split_manifest(manifest_path)
                splits = manifest.get("splits", {})
            elif split_strategy == "fixed_random":
                splits = _build_fixed_random_splits(
                    n_days,
                    train_ratio=config.get("italian_split_train_ratio", 0.70),
                    seed=config.get("italian_split_seed", 42),
                )
            else:
                raise ValueError(
                    "italian_split_strategy must be 'fixed_random' or 'manifest', "
                    f"got {split_strategy!r}"
                )
            if split_name not in splits:
                raise ValueError(
                    f"Italian split {split_name!r} not found for strategy {split_strategy!r}; "
                    f"available splits: {sorted(splits)}"
                )
            days = _normalize_day_indices(splits[split_name], n_days)
    else:
        split_name = "all"
        days = list(range(n_days))
    window_days = max(1, int(window_days))
    if window_days > 1:
        day_set = set(days)
        contiguous_days = [
            day for day in days
            if day + window_days <= n_days
            and all(day + offset in day_set for offset in range(window_days))
        ]
        if contiguous_days:
            days = contiguous_days
        else:
            days = [day for day in days if day + window_days <= n_days]
        if not days:
            raise ValueError(
                f"No valid Italian {window_days}-day window starts for split {split_name!r}"
            )
    day = int(days[int(rng.randint(0, len(days)))])
    return day, split_name, days


def _generate_italian_electric_profiles(config, rng):
    _, data, groups = _load_italian_data(config["italian_data_path"])
    n_agents = config["num_agents"]
    T = config["episode_length"]
    n_rows = data.shape[0]
    multi_day = bool(config.get("multi_day_episode_enable", False))
    day_length = int(config.get("day_boundary_interval", 24)) if multi_day else T
    day_length = max(1, day_length)
    window_days = int(config.get("episode_days", max(1, T // day_length)))
    if n_rows < T:
        raise ValueError(f"Italian profile length {n_rows} is shorter than episode length {T}")
    if multi_day:
        if T != day_length * window_days:
            raise ValueError(
                f"episode_length={T} must equal day_boundary_interval={day_length} "
                f"* episode_days={window_days} in multi-day mode"
            )
        if n_rows % day_length != 0:
            raise ValueError("Italian multi-day sampling requires complete day blocks")
        n_days = n_rows // day_length
        day, split_name, day_pool = _select_italian_day(
            config, rng, n_days, window_days=window_days
        )
        start = day * day_length
    elif n_rows % T == 0:
        window_days = 1
        n_days = n_rows // T
        day, split_name, day_pool = _select_italian_day(config, rng, n_days)
        start = day * T
    else:
        window_days = 1
        if config.get("italian_day_indices") is not None or config.get("italian_split_enable", False):
            raise ValueError("Italian split sampling requires complete day blocks")
        day = None
        split_name = "sliding"
        day_pool = []
        start = int(rng.randint(0, n_rows - T + 1))
    day_slice = slice(start, start + T)
    pv_groups = _split_columns(groups["pv"], n_agents)
    wt_groups = _split_columns(groups["wt"], n_agents)
    load_groups = _split_columns(groups["load_e"], n_agents)
    pv = np.zeros((n_agents, T), dtype=np.float32)
    wt = np.zeros((n_agents, T), dtype=np.float32)
    load_e = np.zeros((n_agents, T), dtype=np.float32)
    for i in range(n_agents):
        pv[i] = _scale_group(data, day_slice, pv_groups[i], float(config["pv_cap"][i]))
        wt[i] = _scale_group(data, day_slice, wt_groups[i], float(config["wt_cap"][i]))
        load_e[i] = _scale_group(data, day_slice, load_groups[i], float(config["load_e_peak"][i]))
    return {
        "pv": pv,
        "wt": wt,
        "load_e": load_e,
        "_italian_day_index": day,
        "_italian_start_day_index": day,
        "_italian_day_indices": (
            list(range(day, day + window_days)) if day is not None else []
        ),
        "_episode_days": window_days,
        "_italian_split_name": split_name,
        "_italian_day_pool": day_pool,
    }


def _derive_heat_from_electric(load_e, config, rng):
    n_agents, T = load_e.shape
    load_h = np.zeros((n_agents, T), dtype=np.float32)
    base_ratio = float(config.get("derived_heat_base_ratio", 0.2))
    variable_ratio = float(config.get("derived_heat_variable_ratio", 0.65))
    for i in range(n_agents):
        peak = float(config["load_h_peak"][i])
        if peak <= 0.0:
            continue
        source = np.asarray(load_e[i], dtype=np.float32)
        lo = float(np.min(source))
        hi = float(np.max(source))
        if hi > lo:
            shape = (source - lo) / (hi - lo)
        else:
            shape = np.zeros(T, dtype=np.float32)
        noise = 1.0 + rng.normal(0, 0.02, size=T)
        load_h[i] = peak * (base_ratio + variable_ratio * shape) * noise
        load_h[i] = np.clip(load_h[i], 0.0, peak * 1.5)
    return load_h.astype(np.float32)


def _generate_synthetic_profiles(config, rng=None):
    """
    生成单日功率曲线。
    Args:
        config: MICROGRID_CONFIG 字典
        rng: numpy RandomState，用于确定性复现（可选）

    Returns:
        dict: {
            "pv":     ndarray [num_agents, T],
            "wt":     ndarray [num_agents, T],
            "load_e": ndarray [num_agents, T],
            "load_h": ndarray [num_agents, T],
        }
    """
    if rng is None:
        rng = np.random.RandomState()

    n_agents = config["num_agents"]
    T = config["episode_length"]
    dt = config["dt"]

    hours = np.arange(T) * dt
    hours_of_day = np.mod(hours, 24.0)

    pv = np.zeros((n_agents, T), dtype=np.float32)
    wt = np.zeros((n_agents, T), dtype=np.float32)
    load_e = np.zeros((n_agents, T), dtype=np.float32)
    load_h = np.zeros((n_agents, T), dtype=np.float32)

    # 日间随机系数（每天一个值，用于增加训练多样性）
    day_factor = rng.uniform(0.9, 1.1)

    for i in range(n_agents):
        # ===== PV 曲线：正弦半波 + 云层扰动 =====
        pv_cap = config["pv_cap"][i]
        if pv_cap > 0:
            cloud_factor = rng.uniform(0.7, 1.0)
            # 日出约6点，日落约18点，中间12小时正常
            pv_raw = np.maximum(0, np.sin(np.pi * (hours_of_day - 6.0) / 12.0))
            # 加入步级微扰动（模拟云层）
            step_noise = 1.0 + rng.uniform(-0.05, 0.05, size=T)
            pv[i] = pv_cap * pv_raw * cloud_factor * step_noise * day_factor
            pv[i] = np.clip(pv[i], 0, pv_cap)

        # ===== WT 曲线：基础风速 + 日变化 + 随机噪声 =====
        wt_cap = config["wt_cap"][i]
        if wt_cap > 0:
            phase = rng.uniform(0, 2 * np.pi)
            base = 0.3 + 0.15 * np.sin(2 * np.pi * hours_of_day / 24.0 + phase)
            noise = rng.normal(0, 0.08, size=T)
            wt_raw = np.clip(base + noise, 0, 1)
            wt[i] = wt_cap * wt_raw * day_factor
            wt[i] = np.clip(wt[i], 0, wt_cap)

        # ===== 电负荷：双峰曲线（早高峰 + 晚高峰） =====
        le_peak = config["load_e_peak"][i]
        if le_peak > 0:
            base_ratio = 0.3
            le_shape = (base_ratio
                        + 0.5 * _gauss(hours_of_day, 9.0, 1.5)
                        + 0.6 * _gauss(hours_of_day, 19.0, 1.5))
            # 加入微扰
            le_noise = 1.0 + rng.normal(0, 0.03, size=T)
            load_e[i] = le_peak * le_shape * le_noise * day_factor
            load_e[i] = np.clip(load_e[i], 0, le_peak * 1.5)

        # ===== 热负荷（仅 MG3/MG4）：早晚用热高峰 =====
        lh_peak = config["load_h_peak"][i]
        if lh_peak > 0:
            base_ratio_h = 0.2
            lh_shape = (base_ratio_h
                        + 0.6 * _gauss(hours_of_day, 7.0, 1.5)
                        + 0.5 * _gauss(hours_of_day, 19.0, 2.0))
            lh_noise = 1.0 + rng.normal(0, 0.03, size=T)
            load_h[i] = lh_peak * lh_shape * lh_noise * day_factor
            load_h[i] = np.clip(load_h[i], 0, lh_peak * 1.5)

    return {
        "pv": pv,
        "wt": wt,
        "load_e": load_e,
        "load_h": load_h,
    }


def generate_daily_profiles(config, rng=None):
    if rng is None:
        rng = np.random.RandomState()
    profiles = _generate_synthetic_profiles(config, rng)
    if config.get("profile_source", "synthetic") == "italian":
        profiles.update(_generate_italian_electric_profiles(config, rng))
        if config.get("derive_heat_from_electric", False):
            profiles["load_h"] = _derive_heat_from_electric(
                profiles["load_e"], config, rng
            )
    return profiles
