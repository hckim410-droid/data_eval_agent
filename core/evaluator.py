from __future__ import annotations

from typing import Any


def _add_check(
    checks: list[dict[str, Any]],
    check_id: str,
    severity: str,
    status: str,
    observed: Any,
    threshold: Any,
    message: str,
) -> None:
    checks.append(
        {
            "check_id": check_id,
            "severity": severity,
            "status": status,
            "observed": observed,
            "threshold": threshold,
            "message": message,
        }
    )


def _decide_overall(checks: list[dict[str, Any]]) -> str:
    has_fail = any(item["status"] == "FAIL" and item["severity"] == "HARD" for item in checks)
    if has_fail:
        return "NO_GO"

    has_warn = any(item["status"] in {"WARN", "FAIL"} for item in checks)
    if has_warn:
        return "CONDITIONAL_GO"

    return "GO"


def _recommendation_text(decision: str) -> str:
    if decision == "GO":
        return "주요 기준을 충족합니다. 다음 단계로 진행하세요."
    if decision == "CONDITIONAL_GO":
        return "경고 항목이 있습니다. 원인 확인 후 개선을 권장합니다."
    return "필수 기준을 충족하지 못했습니다. 데이터 보완이 필요합니다."


def evaluate_o1(profile: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    dataset_meta = profile.get("dataset_meta", {})
    columns_summary = profile.get("columns_summary", {})
    schema = profile.get("schema", [])
    target_summary = profile.get("target_summary", {})

    min_rows = policy.get("min_rows")
    max_target_missing = policy.get("max_target_missing_rate")
    min_usable_features = policy.get("min_usable_features")
    max_duplicate_rate = policy.get("max_duplicate_row_rate")
    minority_warn = policy.get("classification_minority_rate_warn")
    minority_fail = policy.get("classification_minority_rate_fail")
    high_cardinality_threshold = policy.get("high_cardinality_warn_threshold")

    total_rows = dataset_meta.get("total_rows")
    if min_rows is not None and total_rows is not None:
        status = "PASS" if total_rows >= min_rows else "FAIL"
        _add_check(
            checks,
            "row_count",
            "HARD",
            status,
            total_rows,
            min_rows,
            "학습 가능한 최소 행 수 기준",
        )
    else:
        _add_check(
            checks,
            "row_count",
            "HARD",
            "WARN",
            total_rows,
            min_rows,
            "행 수 기준 확인을 위한 정책 또는 데이터 정보가 부족합니다.",
        )

    target_missing_rate = target_summary.get("missing_rate")
    if target_missing_rate is None:
        target_name = dataset_meta.get("target_column")
        target_missing_rate = (
            columns_summary.get(target_name, {}).get("missing_rate") if target_name else None
        )

    if max_target_missing is not None and target_missing_rate is not None:
        status = "PASS" if target_missing_rate <= max_target_missing else "FAIL"
        _add_check(
            checks,
            "target_missing_rate",
            "HARD",
            status,
            round(float(target_missing_rate), 6),
            max_target_missing,
            "타깃 결측 비율 기준",
        )
    else:
        _add_check(
            checks,
            "target_missing_rate",
            "HARD",
            "WARN",
            target_missing_rate,
            max_target_missing,
            "타깃 결측 비율 정보를 확인할 수 없습니다.",
        )

    target_column = dataset_meta.get("target_column")
    usable_features = 0
    for name, summary in columns_summary.items():
        if name == target_column:
            continue
        if summary.get("missing_rate", 1.0) < 1.0:
            usable_features += 1

    if min_usable_features is not None:
        status = "PASS" if usable_features >= min_usable_features else "FAIL"
        _add_check(
            checks,
            "usable_feature_count",
            "HARD",
            status,
            usable_features,
            min_usable_features,
            "사용 가능한 피처 수 기준",
        )
    else:
        _add_check(
            checks,
            "usable_feature_count",
            "HARD",
            "WARN",
            usable_features,
            min_usable_features,
            "피처 수 기준 정책이 없습니다.",
        )

    duplicate_row_rate = dataset_meta.get("duplicate_row_rate")
    if max_duplicate_rate is not None and duplicate_row_rate is not None:
        status = "PASS" if duplicate_row_rate <= max_duplicate_rate else "WARN"
        _add_check(
            checks,
            "duplicate_row_rate",
            "WARN",
            status,
            round(float(duplicate_row_rate), 6),
            max_duplicate_rate,
            "중복 행 비율 기준",
        )
    else:
        _add_check(
            checks,
            "duplicate_row_rate",
            "WARN",
            "WARN",
            duplicate_row_rate,
            max_duplicate_rate,
            "중복 행 비율 정보를 확인할 수 없습니다.",
        )

    task_type = dataset_meta.get("task_type")
    if task_type == "classification":
        top_k = target_summary.get("top_k") or {}
        non_missing = dataset_meta.get("sampled_rows")
        if non_missing is None:
            non_missing = sum(top_k.values()) if top_k else None

        missing_rate = target_summary.get("missing_rate", 0.0)
        if non_missing is not None:
            non_missing = int(non_missing * (1 - float(missing_rate)))

        minority_rate = None
        if top_k and non_missing:
            minority_rate = min(top_k.values()) / max(non_missing, 1)

        if minority_rate is not None and minority_warn is not None and minority_fail is not None:
            if minority_rate < minority_fail:
                status = "FAIL"
            elif minority_rate < minority_warn:
                status = "WARN"
            else:
                status = "PASS"
            message = "소수 클래스 비율 기준 (상위 범주 기준)"
        else:
            status = "WARN"
            message = "소수 클래스 비율 계산에 필요한 정보가 부족합니다."

        _add_check(
            checks,
            "classification_minority_rate",
            "HARD",
            status,
            round(minority_rate, 6) if minority_rate is not None else None,
            {"warn": minority_warn, "fail": minority_fail},
            message,
        )

    if high_cardinality_threshold is not None:
        dtype_map = {item.get("name"): str(item.get("dtype")) for item in schema}
        high_card_cols = []
        for name, summary in columns_summary.items():
            if name == target_column:
                continue
            dtype = dtype_map.get(name, "")
            is_categorical = any(
                marker in dtype for marker in ("object", "category", "bool")
            )
            if not is_categorical:
                continue
            if summary.get("n_unique", 0) >= high_cardinality_threshold:
                high_card_cols.append(name)

        status = "WARN" if high_card_cols else "PASS"
        message = "고카디널리티 범주형 컬럼 기준"
        if high_card_cols:
            message = f"고카디널리티 컬럼: {', '.join(high_card_cols[:5])}"

        _add_check(
            checks,
            "high_cardinality_categorical",
            "WARN",
            status,
            len(high_card_cols),
            high_cardinality_threshold,
            message,
        )

    decision = _decide_overall(checks)
    return {
        "decision": decision,
        "checks": checks,
        "recommendation_text": _recommendation_text(decision),
    }
