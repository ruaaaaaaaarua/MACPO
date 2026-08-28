"""Strict deterministic evaluation on fixed Italian microgrid scenarios."""

from __future__ import annotations

from collections.abc import Iterator, Mapping as MappingABC
from dataclasses import dataclass
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np

from baselines.utils.microgrid_vec_env import MicrogridVecEnv
from envs.microgrid.config import MICROGRID_CONFIG


ITALIAN_DATASET_DAYS = 28
VALIDATION_DAYS = (8, 17, 21, 23)
TEST_DAYS = (1, 7, 14, 24)
# Evaluation deliberately uses one scenario seed per locked day.
DEFAULT_NOISE_SEEDS = (4200,)
TEST_NOISE_SEEDS = (5200,)


@dataclass(frozen=True)
class FixedScenario:
    day: int
    seed: int


def _freeze(value):
    if isinstance(value, MappingABC):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, np.ndarray):
        array = np.asarray(value).view()
        array.setflags(write=False)
        return array
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    return value


@dataclass(frozen=True)
class EvaluationContext(MappingABC):
    """Read-only state exposed to diagnostic contextual policies."""

    episode_step: int
    config: Mapping[str, object]
    profiles: Mapping[str, object]

    def __post_init__(self):
        object.__setattr__(self, "episode_step", int(self.episode_step))
        object.__setattr__(self, "config", _freeze(self.config))
        object.__setattr__(self, "profiles", _freeze(self.profiles))

    def __getitem__(self, key: str):
        if key == "episode_step":
            return self.episode_step
        if key == "config":
            return self.config
        if key == "profiles":
            return self.profiles
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return iter(("episode_step", "config", "profiles"))

    def __len__(self) -> int:
        return 3


def build_scenarios(days: Iterable[int], seeds: Iterable[int]) -> tuple[FixedScenario, ...]:
    return tuple(FixedScenario(int(day), int(seed)) for day in days for seed in seeds)


def load_manifest_splits(path=None) -> Mapping[str, tuple[int, ...]]:
    """Load and validate the locked 28-day train/validation/test manifest."""
    manifest_path = Path(
        path or MICROGRID_CONFIG["italian_split_manifest_path"]
    ).expanduser()
    with manifest_path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)

    declared_days = int(manifest.get("n_days", ITALIAN_DATASET_DAYS))
    if declared_days != ITALIAN_DATASET_DAYS:
        raise ValueError(
            f"Italian manifest must declare {ITALIAN_DATASET_DAYS} days, got {declared_days}"
        )
    raw_splits = manifest.get("splits")
    if not isinstance(raw_splits, dict):
        raise ValueError("Italian manifest must contain a 'splits' object")

    normalized = {}
    for name in ("train", "validation", "test"):
        raw_days = raw_splits.get(name)
        if raw_days is None and name == "validation":
            raw_days = raw_splits.get("val")
        if not isinstance(raw_days, list) or not raw_days:
            raise ValueError(f"Italian manifest split {name!r} must be non-empty")
        days = tuple(int(day) for day in raw_days)
        if len(days) != len(set(days)):
            raise ValueError(f"Italian manifest split {name!r} contains duplicate days")
        invalid = [day for day in days if day < 0 or day >= ITALIAN_DATASET_DAYS]
        if invalid:
            raise ValueError(
                f"Italian manifest split {name!r} has days outside 0..27: {invalid}"
            )
        normalized[name] = days

    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = set(normalized[left]) & set(normalized[right])
        if overlap:
            raise ValueError(
                f"Italian manifest splits {left!r} and {right!r} overlap: {sorted(overlap)}"
            )
    covered = set().union(*(set(days) for days in normalized.values()))
    if covered != set(range(ITALIAN_DATASET_DAYS)):
        raise ValueError("Italian manifest train/validation/test splits must cover all 28 days")
    if normalized["validation"] != VALIDATION_DAYS:
        raise ValueError(
            "Italian manifest validation days do not match the locked fixed days "
            f"{VALIDATION_DAYS}"
        )
    if normalized["test"] != TEST_DAYS:
        raise ValueError(
            f"Italian manifest test days do not match the locked fixed days {TEST_DAYS}"
        )
    return MappingProxyType(normalized)


# Fail at module load rather than silently evaluating against a malformed manifest.
MANIFEST_SPLITS = load_manifest_splits()


def _first_info(infos):
    if isinstance(infos, (list, tuple)) and infos:
        return infos[0] if isinstance(infos[0], dict) else {}
    return infos if isinstance(infos, dict) else {}


def _positive_sum(values) -> float:
    array = np.asarray(values if values is not None else [], dtype=np.float64)
    return float(np.maximum(array, 0.0).sum())


def _lagged_correlation(requested: Sequence[float], h2_load: Sequence[float], lag=4):
    requested_array = np.asarray(requested, dtype=np.float64)
    load_array = np.asarray(h2_load, dtype=np.float64)
    if requested_array.size <= lag or load_array.size <= lag:
        return None
    left = requested_array[:-lag]
    right = load_array[lag:]
    if left.size < 2 or right.size != left.size:
        return None
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        return None
    if float(np.std(left)) <= 1e-12 or float(np.std(right)) <= 1e-12:
        return None
    correlation = float(np.corrcoef(left, right)[0, 1])
    return correlation if math.isfinite(correlation) else None


def _same_time_correlation(left, right):
    left_array = np.asarray(left, dtype=np.float64).reshape(-1)
    right_array = np.asarray(right, dtype=np.float64).reshape(-1)
    if left_array.size < 2 or right_array.size != left_array.size:
        return None
    if not np.isfinite(left_array).all() or not np.isfinite(right_array).all():
        return None
    if float(np.std(left_array)) <= 1e-12 or float(np.std(right_array)) <= 1e-12:
        return None
    correlation = float(np.corrcoef(left_array, right_array)[0, 1])
    return correlation if math.isfinite(correlation) else None


def _mean_optional(values):
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else None


def _scenario_overrides(base_overrides, scenario, split_name):
    overrides = dict(base_overrides)
    overrides.update(
        {
            "italian_split_enable": True,
            "italian_split_strategy": "manifest",
            "italian_split_name": split_name,
            # Manual pinning is allowed only after strict membership validation.
            "italian_day_indices": [scenario.day],
        }
    )
    return overrides


def _validate_scenarios(base_overrides, scenarios, split_name):
    if not scenarios:
        raise ValueError("at least one fixed scenario is required")
    manifest_path = base_overrides.get(
        "italian_split_manifest_path",
        MICROGRID_CONFIG["italian_split_manifest_path"],
    )
    splits = load_manifest_splits(manifest_path)
    if split_name not in splits:
        raise ValueError(
            f"split_name must be one of {tuple(splits)}, got {split_name!r}"
        )
    allowed = set(splits[split_name])
    invalid = sorted({int(item.day) for item in scenarios if int(item.day) not in allowed})
    if invalid:
        raise ValueError(
            f"fixed scenario days {invalid} are not in manifest split {split_name!r}"
        )
    locked_seed = {"train": 30, "validation": 4200, "test": 5200}[split_name]
    invalid_seeds = sorted(
        {int(item.seed) for item in scenarios if int(item.seed) != locked_seed}
    )
    if invalid_seeds:
        raise ValueError(
            f"manifest split {split_name!r} requires scenario seed {locked_seed}; "
            f"got {invalid_seeds}"
        )


def _evaluate_policy_shared(
    contextual_action_fn,
    base_overrides,
    scenarios,
    *,
    algorithm,
    split_name,
):
    _validate_scenarios(base_overrides, scenarios, split_name)

    episodes = []
    saturated_values = 0
    total_action_values = 0
    privileged_diagnostic = bool(
        getattr(contextual_action_fn, "privileged_diagnostic", False)
    )
    for scenario in scenarios:
        overrides = _scenario_overrides(base_overrides, scenario, split_name)
        env = MicrogridVecEnv(num_envs=1, auto_reset=False, config_overrides=overrides)
        try:
            obs_flat, _ = env.reset(seed=scenario.seed)
            raw_env = env.envs[0].env
            episode_return = 0.0
            base_cost = 0.0
            total_cost = 0.0
            external_h2_buy = 0.0
            emergency_h2_buy = 0.0
            emergency_h2_cost = 0.0
            planned_external_buy = 0.0
            planned_external_cost = 0.0
            internal_h2_trade = 0.0
            action_requested_buy = 0.0
            effective_buy = 0.0
            clipped_buy = 0.0
            pending = 0.0
            delivered = 0.0
            transport_shipment_count = 0.0
            transport_gross = 0.0
            transport_loss = 0.0
            transport_delayed_gross = 0.0
            transport_etas = []
            route_counts = np.zeros(3, dtype=np.float64)
            horizon_clipped_buy = 0.0
            edge_utilization_max = 0.0
            requested_series = []
            h2_load_series = []
            arrival_series = []
            arrival_load_series = []
            terminal_h2_shortfall_kg = 0.0
            terminal_h2_settlement_cost = 0.0
            effective_external_h2_cost_yuan_per_kg = 0.0
            low_h2_hits = 0.0
            max_action_abs = 0.0
            terminal_soc_ratios = []
            terminal_h2_ratios = []
            steps = 0
            done = False
            while not done:
                obs = obs_flat.reshape(env.num_agents, env.obs_dim)
                context = EvaluationContext(
                    episode_step=int(raw_env.t),
                    config=raw_env.cfg,
                    profiles=raw_env.profiles,
                )
                raw_action = np.asarray(
                    contextual_action_fn(obs, context), dtype=np.float32
                )
                expected_shape = (env.num_agents, env.action_dim)
                if raw_action.shape != expected_shape:
                    raise ValueError(
                        f"policy returned {raw_action.shape}; expected {expected_shape}"
                    )
                saturated_values += int(np.count_nonzero(np.abs(raw_action) >= 0.95))
                total_action_values += int(raw_action.size)
                action = np.clip(raw_action, -1.0, 1.0)
                max_action_abs = max(max_action_abs, float(np.max(np.abs(action))))
                obs_flat, rewards, terms, truncs, infos = env.step(action)
                episode_return += float(np.asarray(rewards).mean())
                info = _first_info(infos)
                base_cost += float(info.get("base_cost", 0.0))
                total_cost += float(info.get("total_cost", info.get("base_cost", 0.0)))
                external_h2_buy += _positive_sum(info.get("e_h2_ext"))
                # v2: 应急/计划两条外购通道分开计量; 计划量同时并入外购总量
                # (v1 中恒为 0, 老结果数值不变)。
                emergency_h2_buy += _positive_sum(info.get("h2_emergency_buy_energy"))
                emergency_h2_cost += float(info.get("h2_emergency_buy_cost", 0.0))
                planned_step_energy = _positive_sum(
                    info.get("h2_planned_external_order_energy")
                )
                planned_external_buy += planned_step_energy
                planned_external_cost += float(
                    info.get("h2_planned_external_order_cost", 0.0)
                )
                external_h2_buy += planned_step_energy
                internal_h2_trade += float(info.get("h2_market_traded", 0.0))
                requested_step = _positive_sum(
                    info.get("h2_action_requested_buy_quantity")
                )
                action_requested_buy += requested_step
                effective_buy += _positive_sum(
                    info.get("h2_action_effective_buy_quantity")
                )
                clipped_buy += _positive_sum(info.get("h2_buy_clip_amount"))
                pending += float(info.get("pending_h2_energy_total", 0.0))
                delivered += _positive_sum(info.get("delivered_h2_energy"))
                shipments = list(info.get("h2_transport_shipments", []))
                transport_shipment_count += float(len(shipments))
                for shipment in shipments:
                    gross = max(0.0, float(shipment.get("gross_quantity", 0.0)))
                    loss = max(0.0, float(shipment.get("loss_quantity", 0.0)))
                    eta = int(shipment.get("eta", 0))
                    rank = int(shipment.get("route_rank", -1))
                    transport_gross += gross
                    transport_loss += loss
                    transport_etas.append(float(eta))
                    if eta > int(info.get("h2_traffic_min_eta", 4)):
                        transport_delayed_gross += gross
                    if 0 <= rank < route_counts.size:
                        route_counts[rank] += 1.0
                horizon_clipped_buy += _positive_sum(
                    info.get("h2_buy_horizon_clip_amount")
                )
                edge_values = info.get("h2_traffic_edge_utilization", {})
                if isinstance(edge_values, MappingABC) and edge_values:
                    edge_utilization_max = max(
                        edge_utilization_max,
                        max(float(value) for value in edge_values.values()),
                    )
                requested_series.append(requested_step)
                h2_load_series.append(_positive_sum(info.get("e_h2_load")))
                arrivals = np.asarray(
                    info.get("delivered_h2_energy", []), dtype=np.float64
                ).reshape(-1)
                arrival_loads = np.asarray(
                    info.get("e_h2_load", []), dtype=np.float64
                ).reshape(-1)
                if arrivals.size and arrivals.shape == arrival_loads.shape:
                    arrival_series.extend(arrivals.tolist())
                    arrival_load_series.extend(arrival_loads.tolist())
                terminal_h2_shortfall_kg += float(
                    info.get("terminal_h2_shortfall_kg", 0.0)
                )
                terminal_h2_settlement_cost += float(
                    info.get("terminal_h2_settlement_cost", 0.0)
                )
                effective_external_h2_cost_yuan_per_kg = float(
                    info.get("effective_external_h2_cost_yuan_per_kg", 0.0)
                )
                ratios = np.asarray(info.get("h2_level_ratio", []), dtype=np.float64)
                threshold = float(raw_env.cfg.get("h2_low_threshold", 0.0))
                low_h2_hits += float(np.count_nonzero(ratios < threshold))
                terminal_soc_ratios = [float(value) for value in info.get("soc", [])]
                terminal_h2_ratios = [float(value) for value in ratios]
                done = bool(np.any(terms) or np.any(truncs))
                steps += 1
            route_total = float(route_counts.sum())
            if route_total > 0.0:
                probabilities = route_counts[route_counts > 0.0] / route_total
                route_entropy = float(
                    -np.sum(probabilities * np.log(probabilities)) / np.log(3.0)
                )
            else:
                route_entropy = 0.0
            episodes.append(
                {
                    "day": scenario.day,
                    "seed": scenario.seed,
                    "return": episode_return,
                    "base_cost": base_cost,
                    "total_cost": total_cost,
                    "external_h2_buy": external_h2_buy,
                    "emergency_h2_buy": emergency_h2_buy,
                    "emergency_h2_cost": emergency_h2_cost,
                    "planned_external_buy": planned_external_buy,
                    "planned_external_cost": planned_external_cost,
                    "internal_h2_trade": internal_h2_trade,
                    "action_requested_buy": action_requested_buy,
                    "effective_buy": effective_buy,
                    "clipped_buy": clipped_buy,
                    "pending": pending,
                    "delivered": delivered,
                    "transport_shipment_count": transport_shipment_count,
                    "transport_gross": transport_gross,
                    "transport_loss": transport_loss,
                    "transport_eta": float(np.mean(transport_etas)) if transport_etas else 0.0,
                    "transport_delayed_rate": (
                        transport_delayed_gross / transport_gross
                        if transport_gross > 0.0 else 0.0
                    ),
                    "route_counts": route_counts.tolist(),
                    "route_entropy": route_entropy,
                    "horizon_clipped_buy": horizon_clipped_buy,
                    "edge_utilization_max": edge_utilization_max,
                    "terminal_soc_ratios": terminal_soc_ratios,
                    "terminal_h2_ratios": terminal_h2_ratios,
                    "order_vs_t4_load_correlation": _lagged_correlation(
                        requested_series, h2_load_series, lag=4
                    ),
                    "arrival_vs_h2_load_correlation": _same_time_correlation(
                        arrival_series, arrival_load_series
                    ),
                    "terminal_h2_shortfall_kg": terminal_h2_shortfall_kg,
                    "terminal_h2_settlement_cost": terminal_h2_settlement_cost,
                    "effective_external_h2_cost_yuan_per_kg": (
                        effective_external_h2_cost_yuan_per_kg
                    ),
                    "low_h2_hits": low_h2_hits,
                    "action_max_abs": max_action_abs,
                    "steps": steps,
                }
            )
        finally:
            env.close()

    def values(key: str) -> np.ndarray:
        return np.asarray([row[key] for row in episodes], dtype=np.float64)

    terminal_soc_values = [value for row in episodes for value in row["terminal_soc_ratios"]]
    terminal_h2_values = [value for row in episodes for value in row["terminal_h2_ratios"]]
    summary = {
        "return_mean": float(values("return").mean()),
        "return_std": float(values("return").std()),
        "base_cost_mean": float(values("base_cost").mean()),
        "total_cost_mean": float(values("total_cost").mean()),
        "external_h2_buy_mean": float(values("external_h2_buy").mean()),
        "emergency_h2_buy_mean": float(values("emergency_h2_buy").mean()),
        "emergency_h2_cost_mean": float(values("emergency_h2_cost").mean()),
        "planned_external_buy_mean": float(values("planned_external_buy").mean()),
        "planned_external_cost_mean": float(values("planned_external_cost").mean()),
        "internal_h2_trade_mean": float(values("internal_h2_trade").mean()),
        "action_requested_buy_mean": float(values("action_requested_buy").mean()),
        "effective_buy_mean": float(values("effective_buy").mean()),
        "clipped_buy_mean": float(values("clipped_buy").mean()),
        "pending_mean": float(values("pending").mean()),
        "delivered_mean": float(values("delivered").mean()),
        "transport_shipment_count_mean": float(values("transport_shipment_count").mean()),
        "transport_gross_mean": float(values("transport_gross").mean()),
        "transport_loss_mean": float(values("transport_loss").mean()),
        "transport_eta_mean": float(values("transport_eta").mean()),
        "transport_delayed_rate_mean": float(values("transport_delayed_rate").mean()),
        "route_entropy_mean": float(values("route_entropy").mean()),
        "horizon_clipped_buy_mean": float(values("horizon_clipped_buy").mean()),
        "edge_utilization_max_mean": float(values("edge_utilization_max").mean()),
        "terminal_soc_ratio_mean": float(np.mean(terminal_soc_values)),
        "terminal_h2_ratio_mean": float(np.mean(terminal_h2_values)),
        "order_vs_t4_load_correlation_mean": _mean_optional(
            row["order_vs_t4_load_correlation"] for row in episodes
        ),
        "arrival_vs_h2_load_correlation_mean": _mean_optional(
            row["arrival_vs_h2_load_correlation"] for row in episodes
        ),
        "terminal_h2_shortfall_kg_mean": float(
            values("terminal_h2_shortfall_kg").mean()
        ),
        "terminal_h2_settlement_cost_mean": float(
            values("terminal_h2_settlement_cost").mean()
        ),
        "effective_external_h2_cost_yuan_per_kg": float(
            values("effective_external_h2_cost_yuan_per_kg").mean()
        ),
        "low_h2_hits_mean": float(values("low_h2_hits").mean()),
        "action_saturation_rate": float(saturated_values / max(total_action_values, 1)),
    }
    return {
        "algorithm": algorithm,
        "split_name": split_name,
        "privileged_diagnostic": privileged_diagnostic,
        "summary": summary,
        "episodes": episodes,
    }


def evaluate_policy(
    action_fn: Callable[[np.ndarray], np.ndarray],
    base_overrides: Mapping[str, object],
    scenarios: Sequence[FixedScenario],
    *,
    algorithm: str,
    split_name: str,
) -> dict:
    """Evaluate an unchanged learned-policy callable ``action_fn(obs)``."""

    def contextual_adapter(obs, _context):
        return action_fn(obs)

    return _evaluate_policy_shared(
        contextual_adapter,
        base_overrides,
        scenarios,
        algorithm=algorithm,
        split_name=split_name,
    )


def evaluate_contextual_policy(
    action_fn: Callable[[np.ndarray, EvaluationContext], np.ndarray],
    base_overrides: Mapping[str, object],
    scenarios: Sequence[FixedScenario],
    *,
    algorithm: str,
    split_name: str,
) -> dict:
    """Evaluate a diagnostic callable receiving ``(obs, read_only_context)``."""
    return _evaluate_policy_shared(
        action_fn,
        base_overrides,
        scenarios,
        algorithm=algorithm,
        split_name=split_name,
    )


def _json_safe(value):
    if isinstance(value, MappingABC):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def append_evaluation_record(path, payload: Mapping[str, object], *, training_episode: int):
    """Append one self-contained, strict-JSON evaluation point to a JSONL curve."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    row = dict(payload)
    row["training_episode"] = int(training_episode)
    row = _json_safe(row)
    with output.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        )
    return row
