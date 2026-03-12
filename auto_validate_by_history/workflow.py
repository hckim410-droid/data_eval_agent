from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

try:
    from scipy.stats import ks_2samp, moment, skew, wasserstein_distance
except ModuleNotFoundError:
    def wasserstein_distance(u_values, v_values) -> float:
        u = np.sort(np.asarray(u_values, dtype=float))
        v = np.sort(np.asarray(v_values, dtype=float))
        if u.size == 0 or v.size == 0:
            return 0.0
        n = min(u.size, v.size)
        if u.size != n:
            u = np.interp(
                np.linspace(0, u.size - 1, n),
                np.arange(u.size),
                u,
            )
        if v.size != n:
            v = np.interp(
                np.linspace(0, v.size - 1, n),
                np.arange(v.size),
                v,
            )
        return float(np.mean(np.abs(u - v)))

    def ks_2samp(data1, data2) -> tuple[float, float]:
        x = np.sort(np.asarray(data1, dtype=float))
        y = np.sort(np.asarray(data2, dtype=float))
        if x.size == 0 or y.size == 0:
            return 0.0, 1.0
        values = np.sort(np.unique(np.concatenate([x, y])))
        cdf_x = np.searchsorted(x, values, side="right") / x.size
        cdf_y = np.searchsorted(y, values, side="right") / y.size
        statistic = float(np.max(np.abs(cdf_x - cdf_y)))
        p_value = float(max(0.0, 1.0 - statistic))
        return statistic, p_value

    def moment(a, moment: int = 1) -> float:
        arr = np.asarray(a, dtype=float)
        if arr.size == 0:
            return 0.0
        centered = arr - arr.mean()
        return float(np.mean(centered ** moment))

    def skew(a) -> float:
        arr = np.asarray(a, dtype=float)
        if arr.size < 3:
            return 0.0
        m2 = moment(arr, 2)
        if abs(m2) <= 1e-12:
            return 0.0
        m3 = moment(arr, 3)
        return float(m3 / (m2 ** 1.5))


@dataclass
class Rule:
    column: str
    rule_type: str
    threshold: float | None
    center: float | None
    scale: float | None
    precision: float
    recall: float
    fpr: float
    pred: pd.Series


NUMERIC_DISTANCE_METRICS = [
    "EMD",
    "JS_div",
    "KL_div",
    "KS_dist",
    "Cohen_dist",
    "Min",
    "Max",
    "Mean",
    "Median",
    "Count",
    "Sum",
    "Range",
    "Skew",
    "2-moment",
    "3-moment",
    "unique_ratio",
    "complete_ratio",
]


def _resolve_column(df: pd.DataFrame, preferred: str | None, candidates: list[str]) -> str | None:
    if preferred and preferred in df.columns:
        return preferred
    lowered = {col.lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
        found = lowered.get(candidate.lower())
        if found:
            return found
    return None


def _safe_float(value: Any, digits: int = 6) -> float | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return round(float(value), digits)


def _mad(values: pd.Series) -> float:
    if values.empty:
        return 0.0
    med = float(values.median())
    return float((values - med).abs().median())


def _compute_probs(data: pd.Series, n_bins: int = 10) -> tuple[np.ndarray, np.ndarray]:
    hist, edges = np.histogram(np.asarray(data, dtype=float), bins=n_bins)
    probs = hist / max(len(data), 1)
    return edges, probs


def _support_intersection(p: np.ndarray, q: np.ndarray) -> list[tuple[float, float]]:
    return [(px, qx) for px, qx in zip(p, q) if px != 0 and qx != 0]


def _kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.sum(p * np.log(p / q)))


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    midpoint = 0.5 * (p + q)
    return 0.5 * _kl_divergence(p, midpoint) + 0.5 * _kl_divergence(q, midpoint)


def _compute_kl_divergence(train_sample: pd.Series, test_sample: pd.Series, n_bins: int = 10) -> float:
    edges, p = _compute_probs(train_sample, n_bins=n_bins)
    _, q = np.histogram(np.asarray(test_sample, dtype=float), bins=edges)
    q = q / max(len(test_sample), 1)
    overlap = _support_intersection(p, q)
    if not overlap:
        return 0.0
    p_arr = np.array([item[0] for item in overlap], dtype=float)
    q_arr = np.array([item[1] for item in overlap], dtype=float)
    return _kl_divergence(p_arr, q_arr)


def _compute_js_divergence(train_sample: pd.Series, test_sample: pd.Series, n_bins: int = 10) -> float:
    edges, p = _compute_probs(train_sample, n_bins=n_bins)
    _, q = np.histogram(np.asarray(test_sample, dtype=float), bins=edges)
    q = q / max(len(test_sample), 1)
    overlap = _support_intersection(p, q)
    if not overlap:
        return 0.0
    p_arr = np.array([item[0] for item in overlap], dtype=float)
    q_arr = np.array([item[1] for item in overlap], dtype=float)
    return _js_divergence(p_arr, q_arr)


def _cohen_d(x: pd.Series, y: pd.Series) -> float:
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return 0.0
    dof = nx + ny - 2
    if dof <= 0:
        return 0.0
    mu1 = float(np.mean(x))
    mu2 = float(np.mean(y))
    var1 = float(np.var(x, ddof=1))
    var2 = float(np.var(y, ddof=1))
    pooled = np.sqrt(((nx - 1) * var1 + (ny - 1) * var2) / dof)
    return float((mu1 - mu2) / (pooled + 1e-10))


def _compute_numeric_distance_vector(sample_p: pd.Series, sample_q: pd.Series) -> dict[str, float | None]:
    row_count_p = len(sample_p)
    sample_p = sample_p.dropna()
    sample_q = sample_q.dropna()
    if sample_p.empty or sample_q.empty:
        return {metric: None for metric in NUMERIC_DISTANCE_METRICS}

    min_val = float(sample_p.min())
    max_val = float(sample_p.max())
    mean_val = float(sample_p.mean())
    median_val = float(sample_p.median())
    sum_val = float(sample_p.sum())
    range_val = max_val - min_val
    skewness = float(skew(sample_p)) if len(sample_p) > 2 else 0.0
    moment2 = float(moment(sample_p, moment=2)) if len(sample_p) > 1 else 0.0
    moment3 = float(moment(sample_p, moment=3)) if len(sample_p) > 2 else 0.0
    unique_ratio = float(len(pd.unique(sample_p)) / len(sample_p))
    complete_ratio = float(len(sample_p) / row_count_p) if row_count_p else 0.0

    emd = float(wasserstein_distance(sample_p, sample_q))
    js_div = float(_compute_js_divergence(sample_p, sample_q))
    kl_div = float(_compute_kl_divergence(sample_p, sample_q))
    _, ks_p_val = ks_2samp(sample_p, sample_q)
    cohen_dist = float(_cohen_d(sample_p, sample_q))

    values = [
        emd,
        js_div,
        kl_div,
        float(ks_p_val),
        cohen_dist,
        min_val,
        max_val,
        mean_val,
        median_val,
        float(row_count_p),
        sum_val,
        range_val,
        skewness,
        moment2,
        moment3,
        unique_ratio,
        complete_ratio,
    ]
    return {
        metric: _safe_float(value)
        for metric, value in zip(NUMERIC_DISTANCE_METRICS, values)
    }


def prepare_history(
    df: pd.DataFrame,
    group_key: str | None,
    time_col: str | None,
) -> dict[str, Any]:
    resolved_group = _resolve_column(df, group_key, ["customer_id", "Customer ID", "id"])
    resolved_time = _resolve_column(
        df, time_col, ["event_month", "date", "datetime", "timestamp", "event_seq", "time"]
    )
    if not resolved_group:
        return {"error": "group_key ??? ?? ? ????."}
    if not resolved_time:
        return {"error": "time ??? ?? ? ????."}

    working = df.copy()
    parsed_time = pd.to_datetime(working[resolved_time], errors="coerce")
    parse_rate = float(parsed_time.notna().mean())
    if parse_rate < 0.7:
        return {
            "error": "time ??? datetime?? ??? ???? ?????. ?? ??? ??????.",
            "parse_rate": _safe_float(parse_rate),
        }

    working["_avh_time"] = parsed_time
    working = working.dropna(subset=[resolved_group, "_avh_time"])
    if working.empty:
        return {"error": "??? group/time ?? ????."}

    working["_avh_day"] = working["_avh_time"].dt.floor("D")
    unique_days = int(working["_avh_day"].nunique())

    sizes = working.groupby(resolved_group).size()
    min_group_days = int(sizes.min()) if not sizes.empty else 0
    median_group_days = float(sizes.median()) if not sizes.empty else 0.0
    history_ready = unique_days >= 30

    numeric_cols = [
        c
        for c in working.columns
        if c not in {resolved_group, resolved_time, "_avh_time", "_avh_day"}
        and is_numeric_dtype(working[c])
    ]
    if not numeric_cols:
        return {
            "error": "??? ??? ?? Feature ??? ??? ? ????.",
            "history_check": {
                "unique_days": unique_days,
                "required_min_days": 30,
                "history_ready": history_ready,
                "min_group_rows": min_group_days,
                "median_group_rows": _safe_float(median_group_days, 3),
            },
        }

    ordered = working.sort_values([resolved_group, "_avh_time"], kind="mergesort")
    latest = ordered.groupby(resolved_group, dropna=True).tail(1).set_index(resolved_group)
    prev = ordered.groupby(resolved_group, dropna=True).nth(-2)
    if resolved_group in prev.columns:
        prev = prev.set_index(resolved_group)

    feature_rows: list[dict[str, Any]] = []
    baseline_stats: dict[str, dict[str, float]] = {}
    for col in numeric_cols:
        s = pd.to_numeric(ordered[col], errors="coerce").dropna()
        if s.empty:
            continue
        med = float(s.median())
        scale = float(_mad(s) * 1.4826)
        if scale <= 1e-9:
            std_val = float(s.std()) if len(s) > 1 else 0.0
            scale = std_val if std_val > 1e-9 else 1.0
        latest_s = pd.to_numeric(latest[col], errors="coerce").dropna()
        prev_s = pd.to_numeric(prev[col], errors="coerce").dropna()
        delta_med = None
        if not latest_s.empty and not prev_s.empty:
            common_n = min(len(latest_s), len(prev_s))
            if common_n > 0:
                delta_med = float(
                    (latest_s.iloc[:common_n].values - prev_s.iloc[:common_n].values).mean()
                )

        baseline_stats[col] = {"center": med, "scale": scale}
        feature_rows.append(
            {
                "feature": col,
                "mean": _safe_float(float(s.mean())),
                "std": _safe_float(float(s.std()) if len(s) > 1 else 0.0),
                "median": _safe_float(med),
                "mad": _safe_float(_mad(s)),
                "p05": _safe_float(float(s.quantile(0.05))),
                "p95": _safe_float(float(s.quantile(0.95))),
                "latest_mean": _safe_float(float(latest_s.mean()) if not latest_s.empty else np.nan),
                "latest_std": _safe_float(float(latest_s.std()) if len(latest_s) > 1 else 0.0),
                "latest_prev_delta_mean": _safe_float(delta_med),
            }
        )

    distance_rows: list[dict[str, Any]] = []
    distance_history_rows: list[dict[str, Any]] = []
    distance_debug: dict[str, Any] = {
        "total_groups": int(ordered[resolved_group].nunique()),
        "common_group_count": 0,
        "eligible_feature_count": 0,
        "per_feature_valid_group_count": {},
        "history_day_pairs": 0,
    }
    common_groups = latest.index.intersection(prev.index)
    distance_debug["common_group_count"] = int(len(common_groups))
    latest_aligned = latest.loc[common_groups] if len(common_groups) else latest.iloc[0:0]
    prev_aligned = prev.loc[common_groups] if len(common_groups) else prev.iloc[0:0]

    for col in numeric_cols:
        latest_num = pd.to_numeric(latest_aligned[col], errors="coerce")
        prev_num = pd.to_numeric(prev_aligned[col], errors="coerce")
        valid_groups = latest_num.dropna().index.intersection(prev_num.dropna().index)
        distance_debug["per_feature_valid_group_count"][col] = int(len(valid_groups))

    daily_frames: dict[pd.Timestamp, pd.DataFrame] = {}
    for day, day_df in ordered.groupby("_avh_day", dropna=True):
        per_day = (
            day_df.sort_values("_avh_time", kind="mergesort")
            .groupby(resolved_group, dropna=True)
            .tail(1)
            .set_index(resolved_group)
        )
        daily_frames[day] = per_day

    sorted_days = sorted(daily_frames.keys())
    distance_debug["history_day_pairs"] = max(len(sorted_days) - 1, 0)
    for prev_day, curr_day in zip(sorted_days[:-1], sorted_days[1:]):
        prev_frame = daily_frames[prev_day]
        curr_frame = daily_frames[curr_day]
        day_common_groups = curr_frame.index.intersection(prev_frame.index)
        if len(day_common_groups) == 0:
            continue

        for col in numeric_cols:
            curr_num = pd.to_numeric(curr_frame.loc[day_common_groups, col], errors="coerce")
            prev_num = pd.to_numeric(prev_frame.loc[day_common_groups, col], errors="coerce")
            valid_groups = curr_num.dropna().index.intersection(prev_num.dropna().index)
            if len(valid_groups) == 0:
                continue
            distance_vector = _compute_numeric_distance_vector(
                curr_num.loc[valid_groups].reset_index(drop=True),
                prev_num.loc[valid_groups].reset_index(drop=True),
            )
            distance_history_rows.append(
                {
                    "day": str(pd.Timestamp(curr_day).date()),
                    "prev_day": str(pd.Timestamp(prev_day).date()),
                    "feature": col,
                    "group_count": int(len(valid_groups)),
                    **distance_vector,
                }
            )

    if distance_history_rows:
        distance_history_df = pd.DataFrame(distance_history_rows).sort_values(["feature", "day"])
        for col in numeric_cols:
            feature_hist = distance_history_df[distance_history_df["feature"] == col].sort_values("day")
            if feature_hist.empty:
                continue
            for metric in NUMERIC_DISTANCE_METRICS:
                metric_hist = feature_hist[["day", "group_count", metric]].dropna(subset=[metric])
                if metric_hist.empty:
                    continue
                latest_metric_row = metric_hist.iloc[-1]
                hist_values = pd.to_numeric(metric_hist[metric].iloc[:-1], errors="coerce").dropna()
                hist_mean = float(hist_values.mean()) if not hist_values.empty else None
                hist_std = (
                    float(hist_values.std(ddof=1))
                    if len(hist_values) > 1
                    else (0.0 if len(hist_values) == 1 else None)
                )
                current_value = float(latest_metric_row[metric])
                z_score = None
                if hist_mean is not None and hist_std is not None and hist_std > 1e-9:
                    z_score = (current_value - hist_mean) / hist_std
                distance_rows.append(
                    {
                        "feature": col,
                        "metric": metric,
                        "latest_day": latest_metric_row["day"],
                        "group_count": int(latest_metric_row["group_count"]),
                        "current_value": _safe_float(current_value),
                        "hist_mean": _safe_float(hist_mean),
                        "hist_std": _safe_float(hist_std),
                        "z_score": _safe_float(z_score),
                    }
                )
    else:
        distance_history_df = pd.DataFrame(
            columns=["day", "prev_day", "feature", "group_count", *NUMERIC_DISTANCE_METRICS]
        )

    distance_debug["eligible_feature_count"] = int(len({row["feature"] for row in distance_rows}))

    feature_df = pd.DataFrame(feature_rows).sort_values("feature")
    if distance_rows:
        distance_df = pd.DataFrame(distance_rows).sort_values(["feature", "metric"])
    else:
        distance_df = pd.DataFrame(
            columns=["feature", "metric", "latest_day", "group_count", "current_value", "hist_mean", "hist_std", "z_score"]
        )
    return {
        "history_check": {
            "unique_days": unique_days,
            "required_min_days": 30,
            "history_ready": history_ready,
            "min_group_rows": min_group_days,
            "median_group_rows": _safe_float(median_group_days, 3),
            "parse_rate": _safe_float(parse_rate),
        },
        "resolved_group_key": resolved_group,
        "resolved_time_column": resolved_time,
        "working_df": ordered,
        "numeric_cols": numeric_cols,
        "feature_df": feature_df,
        "distance_df": distance_df,
        "distance_history_df": distance_history_df,
        "distance_debug": distance_debug,
        "baseline_stats": baseline_stats,
    }


def generate_synthetic_anomalies(
    prepared: dict[str, Any], random_state: int = 42, per_feature: int = 25
) -> pd.DataFrame:
    working = prepared["working_df"]
    numeric_cols = prepared["numeric_cols"]
    baseline = prepared["baseline_stats"]
    rng = np.random.default_rng(random_state)

    base_rows = working[numeric_cols].dropna(how="all")
    if base_rows.empty:
        return pd.DataFrame()
    sample_n = min(max(per_feature * len(numeric_cols), 80), len(base_rows))
    normal = base_rows.sample(n=sample_n, random_state=random_state).copy()
    normal["__is_anomaly"] = 0
    normal["__anomaly_type"] = "normal"
    normal["__anomaly_feature"] = "-"

    synthetic_rows: list[pd.Series] = []
    for col in numeric_cols:
        stats = baseline.get(col, {})
        center = float(stats.get("center", 0.0))
        scale = float(stats.get("scale", 1.0))
        col_source = pd.to_numeric(base_rows[col], errors="coerce").dropna()
        if col_source.empty:
            continue
        draw_n = min(per_feature, len(col_source))
        sampled = col_source.sample(n=draw_n, random_state=random_state)
        for val in sampled:
            spike = normal.sample(n=1, random_state=int(rng.integers(1_000_000))).iloc[0].copy()
            spike[col] = float(val) + (4.0 * scale)
            spike["__is_anomaly"] = 1
            spike["__anomaly_type"] = "spike"
            spike["__anomaly_feature"] = col
            synthetic_rows.append(spike)

            drop = normal.sample(n=1, random_state=int(rng.integers(1_000_000))).iloc[0].copy()
            drop[col] = float(val) - (4.0 * scale)
            drop["__is_anomaly"] = 1
            drop["__anomaly_type"] = "drop"
            drop["__anomaly_feature"] = col
            synthetic_rows.append(drop)

            miss = normal.sample(n=1, random_state=int(rng.integers(1_000_000))).iloc[0].copy()
            miss[col] = np.nan
            miss["__is_anomaly"] = 1
            miss["__anomaly_type"] = "missing"
            miss["__anomaly_feature"] = col
            synthetic_rows.append(miss)

            shift = normal.sample(n=1, random_state=int(rng.integers(1_000_000))).iloc[0].copy()
            shift[col] = center + (rng.choice([-1.0, 1.0]) * 3.5 * scale)
            shift["__is_anomaly"] = 1
            shift["__anomaly_type"] = "mean_shift"
            shift["__anomaly_feature"] = col
            synthetic_rows.append(shift)

    syn = pd.DataFrame(synthetic_rows)
    if syn.empty:
        return normal
    all_df = pd.concat([normal, syn], ignore_index=True)
    return all_df


def _eval_binary(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    y_t = y_true.astype(int)
    y_p = y_pred.astype(int)
    tp = int(((y_t == 1) & (y_p == 1)).sum())
    tn = int(((y_t == 0) & (y_p == 0)).sum())
    fp = int(((y_t == 0) & (y_p == 1)).sum())
    fn = int(((y_t == 1) & (y_p == 0)).sum())
    precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
    fpr = float(fp / (fp + tn)) if (fp + tn) else 0.0
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "fpr": fpr,
    }


def build_rule_candidates(prepared: dict[str, Any], synthetic_df: pd.DataFrame) -> list[Rule]:
    if synthetic_df.empty:
        return []
    y_true = synthetic_df["__is_anomaly"].astype(int)
    baseline = prepared["baseline_stats"]
    candidates: list[Rule] = []
    z_thresholds = [2.5, 3.0, 3.5, 4.0]
    for col, stats in baseline.items():
        center = float(stats["center"])
        scale = float(stats["scale"])
        series = pd.to_numeric(synthetic_df[col], errors="coerce")
        for z_t in z_thresholds:
            pred = ((series - center).abs() / scale) >= z_t
            metrics = _eval_binary(y_true, pred.fillna(False))
            candidates.append(
                Rule(
                    column=col,
                    rule_type="zscore",
                    threshold=z_t,
                    center=center,
                    scale=scale,
                    precision=metrics["precision"],
                    recall=metrics["recall"],
                    fpr=metrics["fpr"],
                    pred=pred.fillna(False),
                )
            )

        miss_pred = series.isna()
        miss_m = _eval_binary(y_true, miss_pred)
        candidates.append(
            Rule(
                column=col,
                rule_type="missing",
                threshold=None,
                center=center,
                scale=scale,
                precision=miss_m["precision"],
                recall=miss_m["recall"],
                fpr=miss_m["fpr"],
                pred=miss_pred,
            )
        )
    return candidates


def select_rules_with_budget(
    candidates: list[Rule], y_true: pd.Series, budget: float
) -> dict[str, Any]:
    if not candidates:
        return {"selected": [], "metrics": None}
    remaining = sorted(
        candidates,
        key=lambda r: (r.recall, r.precision, -r.fpr),
        reverse=True,
    )
    selected: list[Rule] = []
    combined = pd.Series(False, index=y_true.index)
    best_metrics = _eval_binary(y_true, combined)

    for cand in remaining:
        tentative = combined | cand.pred
        metrics = _eval_binary(y_true, tentative)
        recall_gain = metrics["recall"] - best_metrics["recall"]
        if metrics["fpr"] <= budget and recall_gain > 0.0001:
            selected.append(cand)
            combined = tentative
            best_metrics = metrics

    if not selected:
        viable = [c for c in remaining if c.fpr <= budget]
        if viable:
            first = viable[0]
            selected = [first]
            combined = first.pred
            best_metrics = _eval_binary(y_true, combined)

    rows = []
    for rule in selected:
        rows.append(
            {
                "column": rule.column,
                "rule_type": rule.rule_type,
                "threshold": _safe_float(rule.threshold) if rule.threshold is not None else "-",
                "center": _safe_float(rule.center),
                "scale": _safe_float(rule.scale),
                "single_rule_precision": _safe_float(rule.precision),
                "single_rule_recall": _safe_float(rule.recall),
                "single_rule_fpr": _safe_float(rule.fpr),
            }
        )

    return {
        "selected": selected,
        "selected_df": pd.DataFrame(rows),
        "combined_pred": combined,
        "metrics": {k: _safe_float(v) for k, v in best_metrics.items()},
    }


def apply_selected_rules(
    df: pd.DataFrame,
    selected_rules: list[Rule],
    baseline_stats: dict[str, dict[str, float]],
) -> pd.Series:
    if df.empty or not selected_rules:
        return pd.Series(False, index=df.index)

    pred = pd.Series(False, index=df.index)
    for rule in selected_rules:
        if rule.column not in df.columns:
            continue
        series = pd.to_numeric(df[rule.column], errors="coerce")
        if rule.rule_type == "missing":
            rule_pred = series.isna()
        else:
            stats = baseline_stats.get(rule.column)
            if not stats:
                continue
            center = float(stats["center"])
            scale = float(stats["scale"])
            threshold = float(rule.threshold) if rule.threshold is not None else 3.0
            rule_pred = ((series - center).abs() / scale) >= threshold
        pred = pred | rule_pred.fillna(False)
    return pred


def evaluate_ground_truth(
    df: pd.DataFrame,
    label_col: str,
    selected_rules: list[Rule],
    baseline_stats: dict[str, dict[str, float]],
) -> dict[str, Any]:
    if label_col not in df.columns:
        return {"error": f"?? ?? `{label_col}`?(?) ????."}
    y_true = pd.to_numeric(df[label_col], errors="coerce").fillna(0).astype(int).clip(0, 1)
    y_pred = apply_selected_rules(df, selected_rules, baseline_stats).astype(int)
    metrics = _eval_binary(y_true, y_pred)
    return {
        "metrics": {k: _safe_float(v) for k, v in metrics.items()},
        "result_df": pd.DataFrame(
            {
                "truth": y_true.astype(int),
                "pred": y_pred.astype(int),
            }
        ),
    }
