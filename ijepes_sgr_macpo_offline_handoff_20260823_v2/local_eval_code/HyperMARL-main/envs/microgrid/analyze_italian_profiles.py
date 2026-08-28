from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HYPERMARL_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HYPERMARL_ROOT))

from envs.microgrid.config import MICROGRID_CONFIG
from envs.microgrid.data_generator import _load_italian_data, _scale_group, _split_columns


NUMERIC_FEATURES = [
    "pv_energy",
    "wind_energy",
    "electric_load_energy",
    "heat_load_energy",
    "renewable_energy",
    "renewable_to_electric_load",
    "renewable_to_total_load",
    "electric_net_energy",
    "electric_net_peak",
    "electric_net_min",
    "electric_net_ramp_abs",
    "residual_electric_demand_energy",
    "electric_surplus_energy",
    "pv_cv",
    "wind_cv",
    "electric_load_cv",
    "heat_load_cv",
    "pv_peak",
    "wind_peak",
    "electric_load_peak",
    "heat_load_peak",
]

CLUSTER_FEATURES = [
    "pv_energy",
    "wind_energy",
    "electric_load_energy",
    "heat_load_energy",
    "renewable_to_electric_load",
    "electric_net_peak",
    "electric_net_ramp_abs",
    "residual_electric_demand_energy",
    "electric_surplus_energy",
    "pv_cv",
    "wind_cv",
    "electric_load_cv",
]


def _cv(values):
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    if abs(mean) < 1e-12:
        return 0.0
    return float(np.std(values) / mean)


def _classify(value, low_q, high_q):
    if value <= low_q:
        return "low"
    if value >= high_q:
        return "high"
    return "mid"


def _derive_heat(load_e, config):
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
        load_h[i] = peak * (base_ratio + variable_ratio * shape)
    return load_h.astype(np.float32)


def _day_profiles(data, groups, config, day):
    n_agents = int(config["num_agents"])
    T = int(config["episode_length"])
    day_slice = slice(day * T, (day + 1) * T)
    pv_groups = _split_columns(groups["pv"], n_agents)
    wind_groups = _split_columns(groups["wt"], n_agents)
    load_groups = _split_columns(groups["load_e"], n_agents)
    pv = np.stack([
        _scale_group(data, day_slice, pv_groups[i], float(config["pv_cap"][i]))
        for i in range(n_agents)
    ])
    wind = np.stack([
        _scale_group(data, day_slice, wind_groups[i], float(config["wt_cap"][i]))
        for i in range(n_agents)
    ])
    load_e = np.stack([
        _scale_group(data, day_slice, load_groups[i], float(config["load_e_peak"][i]))
        for i in range(n_agents)
    ])
    load_h = _derive_heat(load_e, config)
    return pv, wind, load_e, load_h


def build_features(config):
    header, data, groups = _load_italian_data(config["italian_data_path"])
    T = int(config["episode_length"])
    if data.shape[0] % T != 0:
        raise ValueError("Italian split analysis requires complete day blocks")
    n_days = data.shape[0] // T
    rows = []
    for day in range(n_days):
        pv, wind, load_e, load_h = _day_profiles(data, groups, config, day)
        pv_total = np.sum(pv, axis=0)
        wind_total = np.sum(wind, axis=0)
        load_e_total = np.sum(load_e, axis=0)
        load_h_total = np.sum(load_h, axis=0)
        renewable_total = pv_total + wind_total
        electric_net = load_e_total - renewable_total
        electric_load_energy = float(np.sum(load_e_total))
        total_load_energy = electric_load_energy + float(np.sum(load_h_total))
        renewable_energy = float(np.sum(renewable_total))
        row = {
            "day": day,
            "pv_energy": float(np.sum(pv_total)),
            "wind_energy": float(np.sum(wind_total)),
            "electric_load_energy": electric_load_energy,
            "heat_load_energy": float(np.sum(load_h_total)),
            "renewable_energy": renewable_energy,
            "renewable_to_electric_load": renewable_energy / max(electric_load_energy, 1e-9),
            "renewable_to_total_load": renewable_energy / max(total_load_energy, 1e-9),
            "electric_net_energy": float(np.sum(electric_net)),
            "electric_net_peak": float(np.max(electric_net)),
            "electric_net_min": float(np.min(electric_net)),
            "electric_net_ramp_abs": float(np.sum(np.abs(np.diff(electric_net)))),
            "residual_electric_demand_energy": float(np.sum(np.maximum(electric_net, 0.0))),
            "electric_surplus_energy": float(np.sum(np.maximum(-electric_net, 0.0))),
            "pv_cv": _cv(pv_total),
            "wind_cv": _cv(wind_total),
            "electric_load_cv": _cv(load_e_total),
            "heat_load_cv": _cv(load_h_total),
            "pv_peak": float(np.max(pv_total)),
            "wind_peak": float(np.max(wind_total)),
            "electric_load_peak": float(np.max(load_e_total)),
            "heat_load_peak": float(np.max(load_h_total)),
        }
        rows.append(row)
    return header, rows


def add_interpretable_labels(rows):
    quantiles = {}
    for key in [
        "pv_energy",
        "wind_energy",
        "electric_load_energy",
        "heat_load_energy",
        "renewable_to_electric_load",
        "electric_net_ramp_abs",
        "residual_electric_demand_energy",
        "electric_surplus_energy",
    ]:
        values = np.asarray([row[key] for row in rows], dtype=float)
        quantiles[key] = np.quantile(values, [1.0 / 3.0, 2.0 / 3.0, 0.75])
    for row in rows:
        labels = []
        pv_class = _classify(row["pv_energy"], quantiles["pv_energy"][0], quantiles["pv_energy"][1])
        wind_class = _classify(row["wind_energy"], quantiles["wind_energy"][0], quantiles["wind_energy"][1])
        load_class = _classify(row["electric_load_energy"], quantiles["electric_load_energy"][0], quantiles["electric_load_energy"][1])
        heat_class = _classify(row["heat_load_energy"], quantiles["heat_load_energy"][0], quantiles["heat_load_energy"][1])
        renewable_class = _classify(row["renewable_to_electric_load"], quantiles["renewable_to_electric_load"][0], quantiles["renewable_to_electric_load"][1])
        ramp_class = _classify(row["electric_net_ramp_abs"], quantiles["electric_net_ramp_abs"][0], quantiles["electric_net_ramp_abs"][1])
        residual_class = _classify(row["residual_electric_demand_energy"], quantiles["residual_electric_demand_energy"][0], quantiles["residual_electric_demand_energy"][1])
        surplus_class = _classify(row["electric_surplus_energy"], quantiles["electric_surplus_energy"][0], quantiles["electric_surplus_energy"][1])
        labels.extend([
            f"pv_{pv_class}",
            f"wind_{wind_class}",
            f"load_{load_class}",
            f"heat_{heat_class}",
            f"renewable_{renewable_class}",
            f"ramp_{ramp_class}",
            f"residual_{residual_class}",
            f"surplus_{surplus_class}",
        ])
        if row["electric_net_ramp_abs"] >= quantiles["electric_net_ramp_abs"][2]:
            labels.append("high_net_ramp_day")
        if row["residual_electric_demand_energy"] >= quantiles["residual_electric_demand_energy"][2]:
            labels.append("high_residual_demand_day")
        if row["electric_surplus_energy"] >= quantiles["electric_surplus_energy"][2]:
            labels.append("high_surplus_day")
        row["pv_class"] = pv_class
        row["wind_class"] = wind_class
        row["load_class"] = load_class
        row["heat_class"] = heat_class
        row["renewable_class"] = renewable_class
        row["ramp_class"] = ramp_class
        row["residual_class"] = residual_class
        row["surplus_class"] = surplus_class
        row["scenario_labels"] = labels
    return quantiles


def standardize_matrix(rows, keys):
    X = np.asarray([[row[key] for key in keys] for row in rows], dtype=float)
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mean) / (std + 1e-9), mean, std


def deterministic_kmeans(Z, k=4, max_iter=100):
    center_indices = [int(np.argmax(np.linalg.norm(Z - np.mean(Z, axis=0), axis=1)))]
    while len(center_indices) < k:
        distances = np.min(
            np.stack([np.linalg.norm(Z - Z[idx], axis=1) for idx in center_indices], axis=1),
            axis=1,
        )
        for idx in center_indices:
            distances[idx] = -1.0
        center_indices.append(int(np.argmax(distances)))
    centers = Z[center_indices].copy()
    labels = np.zeros(Z.shape[0], dtype=np.int64)
    for _ in range(max_iter):
        distances = np.stack([np.linalg.norm(Z - center, axis=1) for center in centers], axis=1)
        next_labels = np.argmin(distances, axis=1)
        next_centers = centers.copy()
        for cluster in range(k):
            mask = next_labels == cluster
            if np.any(mask):
                next_centers[cluster] = np.mean(Z[mask], axis=0)
        if np.array_equal(labels, next_labels):
            centers = next_centers
            break
        labels = next_labels
        centers = next_centers
    return labels, centers, center_indices


def assign_splits(rows, labels, centers, Z):
    for row, label in zip(rows, labels):
        row["cluster"] = int(label)
    stress_keys = {
        "electric_load_energy": 1.0,
        "heat_load_energy": 0.7,
        "electric_net_ramp_abs": 1.0,
        "residual_electric_demand_energy": 1.0,
        "electric_net_peak": 0.5,
        "renewable_to_electric_load": -0.7,
        "wind_cv": 0.3,
    }
    stress_Z, _, _ = standardize_matrix(rows, list(stress_keys))
    weights = np.asarray([stress_keys[key] for key in stress_keys], dtype=float)
    stress = stress_Z @ weights
    validation = []
    test = []
    for cluster in sorted(set(int(x) for x in labels)):
        days = [row["day"] for row in rows if row["cluster"] == cluster]
        distances = {day: float(np.linalg.norm(Z[day] - centers[cluster])) for day in days}
        validation_day = min(days, key=lambda day: (distances[day], day))
        validation.append(validation_day)
        candidates = [day for day in days if day != validation_day]
        if candidates:
            test_day = max(candidates, key=lambda day: (stress[day], -day))
            test.append(test_day)
    remaining = [row["day"] for row in rows if row["day"] not in set(validation + test)]
    while len(validation) < 4 and remaining:
        day = remaining.pop(0)
        validation.append(day)
    while len(test) < 4 and remaining:
        day = max(remaining, key=lambda x: (stress[x], -x))
        remaining.remove(day)
        test.append(day)
    validation = sorted(validation[:4])
    test = sorted(test[:4])
    heldout = set(validation + test)
    train = sorted(row["day"] for row in rows if row["day"] not in heldout)
    for row in rows:
        if row["day"] in train:
            row["split"] = "train"
        elif row["day"] in validation:
            row["split"] = "validation"
        elif row["day"] in test:
            row["split"] = "test"
        else:
            row["split"] = "unused"
        row["stress_score"] = float(stress[row["day"]])
    return {"train": train, "validation": validation, "val": validation, "test": test, "all": sorted(row["day"] for row in rows)}


def split_summary(rows, splits):
    out = {}
    for name, days in splits.items():
        if name == "val":
            continue
        subset = [row for row in rows if row["day"] in set(days)]
        labels = Counter(label for row in subset for label in row["scenario_labels"])
        out[name] = {
            "n_days": len(days),
            "days": days,
            "feature_mean": {key: float(np.mean([row[key] for row in subset])) if subset else 0.0 for key in NUMERIC_FEATURES},
            "label_counts": dict(sorted(labels.items())),
        }
    return out


def write_features_csv(path, rows):
    fieldnames = [
        "day",
        "split",
        "cluster",
        "stress_score",
        *NUMERIC_FEATURES,
        "pv_class",
        "wind_class",
        "load_class",
        "heat_class",
        "renewable_class",
        "ramp_class",
        "residual_class",
        "surplus_class",
        "scenario_labels",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            item = dict(row)
            item["scenario_labels"] = "|".join(row["scenario_labels"])
            writer.writerow({key: item.get(key, "") for key in fieldnames})


def write_manifest(path, source_csv, header, rows, splits, quantiles, labels, centers, initial_center_indices):
    cluster_days = defaultdict(list)
    for row in rows:
        cluster_days[str(row["cluster"])].append(row["day"])
    manifest = {
        "source_csv": str(source_csv),
        "n_hours": len(rows) * int(MICROGRID_CONFIG["episode_length"]),
        "n_days": len(rows),
        "episode_length": int(MICROGRID_CONFIG["episode_length"]),
        "columns": {
            "pv": [name for name in header if name.startswith("Ppv")],
            "wind": [name for name in header if name.startswith("Pw")],
            "electric_load": [name for name in header if name.startswith("PL")],
        },
        "feature_scaling": "environment_capacity_scaled",
        "heat_derivation": {
            "base_ratio": float(MICROGRID_CONFIG.get("derived_heat_base_ratio", 0.0)),
            "variable_ratio": float(MICROGRID_CONFIG.get("derived_heat_variable_ratio", 0.0)),
            "noise_for_analysis": 0.0,
        },
        "cluster_features": CLUSTER_FEATURES,
        "split_algorithm": "deterministic_kmeans_k4_then_one_representative_validation_and_one_high_stress_test_day_per_cluster",
        "initial_center_indices": [int(x) for x in initial_center_indices],
        "splits": splits,
        "cluster_days": {key: sorted(value) for key, value in sorted(cluster_days.items())},
        "quantiles": {key: [float(x) for x in value] for key, value in quantiles.items()},
        "summary": split_summary(rows, splits),
        "days": {
            str(row["day"]): {
                "split": row["split"],
                "cluster": int(row["cluster"]),
                "stress_score": float(row["stress_score"]),
                "labels": row["scenario_labels"],
                "features": {key: float(row[key]) for key in NUMERIC_FEATURES},
            }
            for row in rows
        },
    }
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=MICROGRID_CONFIG["italian_data_path"])
    parser.add_argument("--features_out", default=str(Path(__file__).with_name("italian_day_features.csv")))
    parser.add_argument("--manifest_out", default=str(Path(__file__).with_name("italian_day_splits.json")))
    args = parser.parse_args()
    config = dict(MICROGRID_CONFIG)
    config["italian_data_path"] = args.csv
    header, rows = build_features(config)
    quantiles = add_interpretable_labels(rows)
    Z, _, _ = standardize_matrix(rows, CLUSTER_FEATURES)
    labels, centers, initial_center_indices = deterministic_kmeans(Z, k=4)
    splits = assign_splits(rows, labels, centers, Z)
    write_features_csv(args.features_out, rows)
    write_manifest(args.manifest_out, args.csv, header, rows, splits, quantiles, labels, centers, initial_center_indices)
    print(f"wrote {args.features_out}")
    print(f"wrote {args.manifest_out}")
    print(json.dumps({key: value for key, value in splits.items() if key != "val"}, sort_keys=True))


if __name__ == "__main__":
    main()
