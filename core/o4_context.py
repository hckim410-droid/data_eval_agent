from __future__ import annotations

from typing import Iterable

import pandas as pd


def _select_columns(df: pd.DataFrame, candidates: Iterable[str]) -> list[str]:
    return [col for col in candidates if col in df.columns]


def _truncate_rows(df: pd.DataFrame, max_rows: int) -> tuple[pd.DataFrame, bool]:
    if max_rows <= 0 or len(df) <= max_rows:
        return df, False
    return df.head(max_rows), True


def build_ontology_context(
    nodes_df: pd.DataFrame, edges_df: pd.DataFrame, max_rows_each: int = 200
) -> str:
    node_cols = _select_columns(
        nodes_df,
        ["node_id", "node_type", "customer_id", "event_month", "properties"],
    )
    edge_cols = _select_columns(
        edges_df,
        ["src_id", "rel_type", "dst_id", "event_month", "properties"],
    )

    nodes_view = nodes_df[node_cols] if node_cols else nodes_df
    edges_view = edges_df[edge_cols] if edge_cols else edges_df

    nodes_trimmed, nodes_truncated = _truncate_rows(nodes_view, max_rows_each)
    edges_trimmed, edges_truncated = _truncate_rows(edges_view, max_rows_each)

    lines = ["[NODES]"]
    lines.append(nodes_trimmed.to_csv(index=False))
    if nodes_truncated:
        lines.append(f"(truncated to {max_rows_each} rows)")

    lines.append("[EDGES]")
    lines.append(edges_trimmed.to_csv(index=False))
    if edges_truncated:
        lines.append(f"(truncated to {max_rows_each} rows)")

    return "\n".join(lines).strip()


def build_general_table_context(
    table_df: pd.DataFrame, max_rows: int = 200
) -> str:
    trimmed, truncated = _truncate_rows(table_df, max_rows)
    body = trimmed.to_csv(index=False)
    if truncated:
        body = f"{body}\n(truncated to {max_rows} rows)"
    return body.strip()
