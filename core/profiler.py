from __future__ import annotations

import time
from typing import Any

import pandas as pd
from pandas.api.types import is_bool_dtype, is_categorical_dtype, is_numeric_dtype, is_object_dtype


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

    if target_col not in working_df.columns:
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
