from __future__ import annotations

from typing import Any

import pandas as pd
from pandas.api.types import is_numeric_dtype

AVH_EVALUATION_CRITERIA: dict[str, Any] = {
    "required_group_key": True,
    "required_time_column": True,
    "min_history_length_per_group": 3,
    "datetime_parse_success_threshold": 0.9,
    "min_usable_history_groups": 30,
    "min_history_ready_rate": 0.7,
    "max_group_size_cv": 2.5,
    "max_duplicate_group_time_rate": 0.01,
    "max_non_monotonic_group_rate": 0.05,
    "max_null_share_in_latest_step": 0.2,
    "max_numeric_shift_score": 4.0,
    "max_target_flip_rate": 0.5,
}


def get_avh_evaluation_criteria() -> dict[str, Any]:
    return dict(AVH_EVALUATION_CRITERIA)


def _resolve_column(df: pd.DataFrame, preferred: str | None, candidates: list[str]) -> str | None:
    if preferred and preferred in df.columns:
        return preferred

    lowered = {col.lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
        match = lowered.get(candidate.lower())
        if match:
            return match
    return None


def _safe_round(value: float | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _mad(values: pd.Series) -> float:
    if values.empty:
        return 0.0
    median = float(values.median())
    return float((values - median).abs().median())


def _compute_numeric_shift_score(
    ordered: pd.DataFrame,
    resolved_group_key: str,
    numeric_cols: list[str],
) -> tuple[float | None, dict[str, float]]:
    if not numeric_cols:
        return None, {}

    per_feature_score: dict[str, float] = {}
    for col in numeric_cols:
        baseline: list[float] = []
        latest: list[float] = []

        for _, group_df in ordered.groupby(resolved_group_key, dropna=True):
            if len(group_df) < 2:
                continue

            numeric_series = pd.to_numeric(group_df[col], errors="coerce")
            diffs = numeric_series.diff().abs().dropna()
            if not diffs.empty:
                baseline.extend(float(v) for v in diffs.values)

            prev_val = pd.to_numeric(group_df.iloc[-2][col], errors="coerce")
            last_val = pd.to_numeric(group_df.iloc[-1][col], errors="coerce")
            if pd.isna(prev_val) or pd.isna(last_val):
                continue
            latest.append(float(abs(last_val - prev_val)))

        if not baseline or not latest:
            continue

        baseline_s = pd.Series(baseline)
        latest_s = pd.Series(latest)
        baseline_median = float(baseline_s.median())
        baseline_mad = _mad(baseline_s)
        robust_scale = baseline_mad if baseline_mad > 1e-12 else float(baseline_s.std())
        if robust_scale <= 1e-12:
            robust_scale = 1.0

        score = (float(latest_s.median()) - baseline_median) / robust_scale
        per_feature_score[col] = _safe_round(max(score, 0.0), 6) or 0.0

    if not per_feature_score:
        return None, {}
    return max(per_feature_score.values()), per_feature_score


def run_auto_validate_by_history(
    df: pd.DataFrame,
    group_key: str | None,
    time_col: str | None,
    target_col: str | None,
) -> dict[str, Any]:
    criteria = get_avh_evaluation_criteria()
    resolved_group_key = _resolve_column(
        df, group_key, ["customer_id", "Customer ID", "customer id", "id"]
    )
    resolved_time_col = _resolve_column(
        df,
        time_col,
        ["event_seq", "event_month", "timestamp", "date", "datetime", "time"],
    )

    summary: dict[str, Any] = {
        "engine": "auto_validate_by_history.streamlit_runner",
        "resolved_group_key": resolved_group_key,
        "resolved_time_column": resolved_time_col,
        "total_rows": int(len(df)),
        "datetime_parse_success_rate": None,
        "total_groups": 0,
        "usable_history_groups": 0,
        "history_ready_rate": None,
        "avg_history_length": None,
        "median_history_length": None,
        "group_size_cv": None,
        "duplicate_group_time_rate": None,
        "non_monotonic_group_rate": None,
        "sequence_monotonic_rate": None,
        "latest_step_null_share": None,
        "latest_step_group_count": 0,
        "numeric_shift_score": None,
        "numeric_shift_by_feature": {},
        "top_numeric_delta": {},
        "target_last_step_change_rate": None,
        "criteria": criteria,
    }

    if not resolved_group_key:
        summary["error"] = "group_key is required for Auto Validate by History"
        return summary
    if not resolved_time_col:
        summary["error"] = "time_column is required for Auto Validate by History"
        return summary

    columns = [resolved_group_key, resolved_time_col] + [
        c for c in df.columns if c not in {resolved_group_key, resolved_time_col}
    ]
    working = df[columns].copy()

    time_series = pd.to_datetime(working[resolved_time_col], errors="coerce")
    parse_success_rate = float(time_series.notna().mean())
    summary["datetime_parse_success_rate"] = _safe_round(parse_success_rate)
    if parse_success_rate >= float(criteria["datetime_parse_success_threshold"]):
        working["_avh_time"] = time_series
    else:
        numeric_time = pd.to_numeric(working[resolved_time_col], errors="coerce")
        working["_avh_time"] = numeric_time

    working = working.dropna(subset=[resolved_group_key, "_avh_time"])
    if working.empty:
        summary["error"] = "no valid rows after resolving group_key/time_column"
        return summary

    grouped = working.groupby(resolved_group_key, dropna=True)
    sizes = grouped.size()
    total_groups = int(len(sizes))
    min_hist_len = int(criteria["min_history_length_per_group"])
    usable_groups = int((sizes >= min_hist_len).sum())
    summary["total_groups"] = total_groups
    summary["usable_history_groups"] = usable_groups
    summary["history_ready_rate"] = (
        round(float(usable_groups / total_groups), 6) if total_groups else None
    )
    summary["avg_history_length"] = round(float(sizes.mean()), 4) if total_groups else None
    summary["median_history_length"] = float(sizes.median()) if total_groups else None
    if total_groups > 1 and float(sizes.mean()) > 0:
        summary["group_size_cv"] = _safe_round(float(sizes.std() / sizes.mean()))

    duplicate_rows = working.duplicated(subset=[resolved_group_key, "_avh_time"]).mean()
    summary["duplicate_group_time_rate"] = _safe_round(float(duplicate_rows))

    monotonic_ok = 0
    monotonic_total = 0
    non_monotonic_count = 0
    for _, group_df in grouped:
        if len(group_df) < min_hist_len:
            continue
        monotonic_total += 1
        if bool(group_df["_avh_time"].is_monotonic_increasing):
            monotonic_ok += 1
        else:
            non_monotonic_count += 1
    if monotonic_total:
        summary["sequence_monotonic_rate"] = round(float(monotonic_ok / monotonic_total), 6)
        summary["non_monotonic_group_rate"] = _safe_round(
            float(non_monotonic_count / monotonic_total)
        )

    numeric_cols = [
        col
        for col in working.columns
        if col not in {resolved_group_key, resolved_time_col, "_avh_time", target_col}
        and is_numeric_dtype(working[col])
    ]
    if usable_groups:
        feature_cols = [
            col
            for col in working.columns
            if col not in {resolved_group_key, resolved_time_col, "_avh_time"}
        ]

        ordered = working.sort_values([resolved_group_key, "_avh_time"], kind="mergesort")
        latest_rows = ordered.groupby(resolved_group_key, dropna=True).tail(1)
        latest_rows = latest_rows.groupby(resolved_group_key, dropna=True).tail(1)
        latest_rows = latest_rows[latest_rows[resolved_group_key].notna()]
        latest_group_count = int(len(latest_rows))
        summary["latest_step_group_count"] = latest_group_count
        if feature_cols and latest_group_count:
            summary["latest_step_null_share"] = _safe_round(
                float(latest_rows[feature_cols].isna().mean().mean())
            )

    if numeric_cols and usable_groups:
        deltas: dict[str, list[float]] = {col: [] for col in numeric_cols}
        ordered = working.sort_values([resolved_group_key, "_avh_time"], kind="mergesort")
        for _, group_df in ordered.groupby(resolved_group_key, dropna=True):
            if len(group_df) < min_hist_len:
                continue
            prev_row = group_df.iloc[-2]
            last_row = group_df.iloc[-1]
            for col in numeric_cols:
                prev_val = pd.to_numeric(prev_row[col], errors="coerce")
                last_val = pd.to_numeric(last_row[col], errors="coerce")
                if pd.isna(prev_val) or pd.isna(last_val):
                    continue
                deltas[col].append(float(abs(last_val - prev_val)))
        mean_deltas = {
            col: round(float(pd.Series(values).mean()), 6)
            for col, values in deltas.items()
            if values
        }
        summary["top_numeric_delta"] = dict(
            sorted(mean_deltas.items(), key=lambda x: x[1], reverse=True)[:5]
        )
        shift_score, shift_detail = _compute_numeric_shift_score(
            ordered=ordered,
            resolved_group_key=resolved_group_key,
            numeric_cols=numeric_cols,
        )
        summary["numeric_shift_score"] = _safe_round(shift_score)
        summary["numeric_shift_by_feature"] = shift_detail

    if target_col and target_col in working.columns and usable_groups:
        changed = 0
        total = 0
        ordered = working.sort_values([resolved_group_key, "_avh_time"], kind="mergesort")
        for _, group_df in ordered.groupby(resolved_group_key, dropna=True):
            if len(group_df) < min_hist_len:
                continue
            prev_val = group_df.iloc[-2][target_col]
            last_val = group_df.iloc[-1][target_col]
            if pd.isna(prev_val) or pd.isna(last_val):
                continue
            total += 1
            if str(prev_val) != str(last_val):
                changed += 1
        if total:
            summary["target_last_step_change_rate"] = round(float(changed / total), 6)

    return summary
