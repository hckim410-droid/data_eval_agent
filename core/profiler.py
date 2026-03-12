from __future__ import annotations

import time
from typing import Any

import pandas as pd
from pandas.api.types import is_bool_dtype, is_categorical_dtype, is_numeric_dtype, is_object_dtype

try:
    from auto_validate_by_history.streamlit_runner import (
        run_auto_validate_by_history as _run_auto_validate_by_history,
    )
except Exception:
    _run_auto_validate_by_history = None


def _build_numeric_stats(series: pd.Series) -> dict[str, Any] | None:
    values = series.dropna()
    if values.empty:
        return None

    return {
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std()),
    }


def _build_top_k(series: pd.Series, top_k: int = 10) -> dict[str, int] | None:
    values = series.dropna()
    if values.empty:
        return None

    counts = values.value_counts(sort=False).nlargest(top_k)
    return {str(index): int(count) for index, count in counts.items()}


def _summarize_column(series: pd.Series) -> dict[str, Any]:
    missing_rate = float(series.isna().mean())
    n_unique = int(series.nunique(dropna=True))
    numeric_stats = _build_numeric_stats(series) if is_numeric_dtype(series) else None

    is_categorical = (
        is_object_dtype(series) or is_categorical_dtype(series) or is_bool_dtype(series)
    )
    top_k = _build_top_k(series) if is_categorical else None

    return {
        "missing_rate": missing_rate,
        "n_unique": n_unique,
        "numeric": numeric_stats,
        "top_k": top_k,
    }


def _apply_sampling(
    df: pd.DataFrame, sampling_mode: str, sample_n: int
) -> tuple[pd.DataFrame, bool]:
    if df.empty:
        return df, False

    safe_sample_n = max(int(sample_n), 1)

    if sampling_mode == "full":
        return df, False

    if sampling_mode == "sample":
        if len(df) <= safe_sample_n:
            return df, False
        return df.sample(n=safe_sample_n, random_state=42), True

    if sampling_mode == "auto":
        if len(df) > 500_000 and len(df) > safe_sample_n:
            return df.sample(n=safe_sample_n, random_state=42), True
        return df, False

    return df, False


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


def _build_avh_summary(
    df: pd.DataFrame, group_key: str | None, time_col: str | None, target_col: str | None
) -> dict[str, Any]:
    if _run_auto_validate_by_history is not None:
        try:
            return _run_auto_validate_by_history(
                df=df,
                group_key=group_key,
                time_col=time_col,
                target_col=target_col,
            )
        except Exception as exc:
            fallback = {
                "engine": "core.profiler._build_avh_summary",
                "external_runner_error": str(exc),
            }
            fallback.update(
                _build_avh_summary_fallback(
                    df=df,
                    group_key=group_key,
                    time_col=time_col,
                    target_col=target_col,
                )
            )
            return fallback

    fallback = {"engine": "core.profiler._build_avh_summary"}
    fallback.update(
        _build_avh_summary_fallback(
            df=df,
            group_key=group_key,
            time_col=time_col,
            target_col=target_col,
        )
    )
    return fallback


def _build_avh_summary_fallback(
    df: pd.DataFrame, group_key: str | None, time_col: str | None, target_col: str | None
) -> dict[str, Any]:
    min_hist_len = 3
    parse_threshold = 0.9
    resolved_group_key = _resolve_column(
        df, group_key, ["customer_id", "Customer ID", "customer id", "id"]
    )
    resolved_time_col = _resolve_column(
        df,
        time_col,
        ["event_seq", "event_month", "timestamp", "date", "datetime", "time"],
    )

    summary: dict[str, Any] = {
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
    }

    if not resolved_group_key:
        summary["error"] = "group_key is required for Auto Validate by History"
        return summary
    if not resolved_time_col:
        summary["error"] = "time_column is required for Auto Validate by History"
        return summary

    working = df[[resolved_group_key, resolved_time_col] + [c for c in df.columns if c not in {resolved_group_key, resolved_time_col}]].copy()
    time_series = pd.to_datetime(working[resolved_time_col], errors="coerce")
    summary["datetime_parse_success_rate"] = round(float(time_series.notna().mean()), 6)
    if time_series.notna().mean() >= parse_threshold:
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
    usable_groups = int((sizes >= min_hist_len).sum())
    summary["total_groups"] = total_groups
    summary["usable_history_groups"] = usable_groups
    summary["history_ready_rate"] = (
        round(float(usable_groups / total_groups), 6) if total_groups else None
    )
    summary["avg_history_length"] = round(float(sizes.mean()), 4) if total_groups else None
    summary["median_history_length"] = (
        float(sizes.median()) if total_groups else None
    )
    if total_groups > 1 and float(sizes.mean()) > 0:
        summary["group_size_cv"] = round(float(sizes.std() / sizes.mean()), 6)

    summary["duplicate_group_time_rate"] = round(
        float(working.duplicated(subset=[resolved_group_key, "_avh_time"]).mean()), 6
    )

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
        summary["non_monotonic_group_rate"] = round(
            float(non_monotonic_count / monotonic_total), 6
        )

    numeric_cols = [
        col
        for col in working.columns
        if col not in {resolved_group_key, resolved_time_col, "_avh_time", target_col}
        and is_numeric_dtype(working[col])
    ]
    if usable_groups:
        ordered = working.sort_values([resolved_group_key, "_avh_time"], kind="mergesort")
        latest_rows = ordered.groupby(resolved_group_key, dropna=True).tail(1)
        latest_rows = latest_rows[latest_rows[resolved_group_key].notna()]
        summary["latest_step_group_count"] = int(len(latest_rows))
        feature_cols = [
            col
            for col in working.columns
            if col not in {resolved_group_key, resolved_time_col, "_avh_time"}
        ]
        if feature_cols and not latest_rows.empty:
            summary["latest_step_null_share"] = round(
                float(latest_rows[feature_cols].isna().mean().mean()), 6
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
        top_deltas = dict(
            sorted(mean_deltas.items(), key=lambda x: x[1], reverse=True)[:5]
        )
        summary["top_numeric_delta"] = top_deltas

    return summary


def build_profile_o1(
    df: pd.DataFrame,
    task_type: str,
    target_col: str,
    group_key: str | None = None,
    time_col: str | None = None,
    sampling_mode: str = "auto",
    sample_n: int = 5000,
) -> dict[str, Any]:
    start_ts = time.perf_counter()
    working_df, sampled = _apply_sampling(df, sampling_mode, sample_n)
    duplicate_row_rate = float(df.duplicated().mean()) if len(df) else 0.0

    dataset_meta = {
        "total_rows": int(len(df)),
        "total_columns": int(len(df.columns)),
        "sampled_rows": int(len(working_df)),
        "sampling_mode": sampling_mode,
        "sample_applied": sampled,
        "duplicate_row_rate": duplicate_row_rate,
        "task_type": task_type,
        "target_column": target_col,
        "group_key": group_key,
        "time_column": time_col,
    }

    schema = [{"name": col, "dtype": str(dtype)} for col, dtype in working_df.dtypes.items()]

    columns_summary = {
        col: _summarize_column(working_df[col]) for col in working_df.columns
    }

    if task_type == "Auto Validate by History":
        target_summary = {
            "mode": "avh",
            "avh_summary": _build_avh_summary(working_df, group_key, time_col, target_col),
        }
    elif target_col not in working_df.columns:
        target_summary = {
            "error": f"target column '{target_col}' not found in data",
        }
    else:
        target_series = working_df[target_col]
        if task_type == "classification":
            target_summary = {
                "missing_rate": float(target_series.isna().mean()),
                "n_unique": int(target_series.nunique(dropna=True)),
                "top_k": _build_top_k(target_series),
            }
        elif task_type == "regression":
            target_summary = {
                "missing_rate": float(target_series.isna().mean()),
                "numeric": _build_numeric_stats(target_series),
            }
        else:
            target_summary = {"error": f"unsupported task_type '{task_type}'"}

    dataset_meta["profiling_seconds"] = round(time.perf_counter() - start_ts, 3)

    return {
        "dataset_meta": dataset_meta,
        "schema": schema,
        "columns_summary": columns_summary,
        "target_summary": target_summary,
    }
