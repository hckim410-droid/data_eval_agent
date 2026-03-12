from __future__ import annotations

from typing import Any

try:
    from auto_validate_by_history.streamlit_runner import (
        get_avh_evaluation_criteria as _get_avh_evaluation_criteria,
    )
except Exception:
    _get_avh_evaluation_criteria = None


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
        return "주요 기준을 충족했습니다. 다음 단계로 진행하세요."
    if decision == "CONDITIONAL_GO":
        return "경고 항목이 있습니다. 원인 확인 및 개선을 권장합니다."
    return "필수 기준을 충족하지 못했습니다. 데이터 보완이 필요합니다."


def _get_avh_criteria() -> dict[str, Any]:
    if _get_avh_evaluation_criteria is not None:
        try:
            return _get_avh_evaluation_criteria()
        except Exception:
            pass

    return {
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


def _evaluate_auto_validate_by_history(
    profile: dict[str, Any], policy: dict[str, Any]
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    target_summary = profile.get("target_summary", {})
    avh_summary = target_summary.get("avh_summary", {})
    criteria = _get_avh_criteria()

    resolved_group_key = avh_summary.get("resolved_group_key")
    resolved_time_col = avh_summary.get("resolved_time_column")
    parse_success_rate = avh_summary.get("datetime_parse_success_rate")
    usable_groups = avh_summary.get("usable_history_groups")
    history_ready_rate = avh_summary.get("history_ready_rate")
    group_size_cv = avh_summary.get("group_size_cv")
    duplicate_rate = avh_summary.get("duplicate_group_time_rate")
    non_monotonic_rate = avh_summary.get("non_monotonic_group_rate")
    monotonic_rate = avh_summary.get("sequence_monotonic_rate")
    latest_step_null_share = avh_summary.get("latest_step_null_share")
    numeric_shift_score = avh_summary.get("numeric_shift_score")
    target_flip_rate = avh_summary.get("target_last_step_change_rate")

    _add_check(
        checks,
        "avh_group_key_resolved",
        "HARD",
        "PASS" if resolved_group_key else "FAIL",
        resolved_group_key,
        "required",
        "Auto Validate by History group key 해석 가능 여부",
    )
    _add_check(
        checks,
        "avh_time_column_resolved",
        "HARD",
        "PASS" if resolved_time_col else "FAIL",
        resolved_time_col,
        "required",
        "Auto Validate by History time column 해석 가능 여부",
    )

    parse_threshold = float(criteria.get("datetime_parse_success_threshold", 0.9))
    if parse_success_rate is None:
        parse_status = "WARN"
    elif float(parse_success_rate) >= parse_threshold:
        parse_status = "PASS"
    else:
        parse_status = "FAIL"
    _add_check(
        checks,
        "avh_datetime_parse_success_rate",
        "HARD",
        parse_status,
        parse_success_rate,
        parse_threshold,
        "시간 컬럼 datetime 파싱 성공 비율",
    )

    min_usable_groups = int(criteria.get("min_usable_history_groups", 1))
    if usable_groups is not None:
        usable_status = "PASS" if int(usable_groups) >= min_usable_groups else "FAIL"
    else:
        usable_status = "WARN"
    _add_check(
        checks,
        "avh_usable_history_groups",
        "HARD",
        usable_status,
        usable_groups,
        min_usable_groups,
        "히스토리 길이 기준을 만족하는 그룹 수",
    )

    ready_threshold = float(criteria.get("min_history_ready_rate", 0.7))
    if history_ready_rate is None:
        ready_status = "WARN"
    elif float(history_ready_rate) >= ready_threshold:
        ready_status = "PASS"
    elif float(history_ready_rate) >= max(ready_threshold - 0.15, 0.0):
        ready_status = "WARN"
    else:
        ready_status = "FAIL"
    _add_check(
        checks,
        "avh_history_ready_rate",
        "WARN",
        ready_status,
        history_ready_rate,
        {"pass": ready_threshold, "warn": max(ready_threshold - 0.15, 0.0)},
        "이력 준비율(usable_history_groups / total_groups)",
    )

    cv_threshold = float(criteria.get("max_group_size_cv", 2.5))
    if group_size_cv is None:
        cv_status = "WARN"
    elif float(group_size_cv) <= cv_threshold:
        cv_status = "PASS"
    else:
        cv_status = "WARN"
    _add_check(
        checks,
        "avh_group_size_cv",
        "WARN",
        cv_status,
        group_size_cv,
        cv_threshold,
        "그룹별 히스토리 길이 변동계수(CV)",
    )

    duplicate_threshold = float(criteria.get("max_duplicate_group_time_rate", 0.01))
    if duplicate_rate is None:
        duplicate_status = "WARN"
    elif float(duplicate_rate) <= duplicate_threshold:
        duplicate_status = "PASS"
    else:
        duplicate_status = "FAIL"
    _add_check(
        checks,
        "avh_duplicate_group_time_rate",
        "HARD",
        duplicate_status,
        duplicate_rate,
        duplicate_threshold,
        "(group_key, time) 중복 비율",
    )

    non_monotonic_threshold = float(criteria.get("max_non_monotonic_group_rate", 0.05))
    if non_monotonic_rate is None:
        non_monotonic_status = "WARN"
    elif float(non_monotonic_rate) <= non_monotonic_threshold:
        non_monotonic_status = "PASS"
    else:
        non_monotonic_status = "FAIL"
    _add_check(
        checks,
        "avh_non_monotonic_group_rate",
        "HARD",
        non_monotonic_status,
        non_monotonic_rate,
        non_monotonic_threshold,
        "시간 순서 비단조 그룹 비율",
    )

    if monotonic_rate is None:
        monotonic_status = "WARN"
    elif float(monotonic_rate) >= 1.0:
        monotonic_status = "PASS"
    else:
        monotonic_status = "WARN"
    _add_check(
        checks,
        "avh_sequence_monotonic_rate",
        "WARN",
        monotonic_status,
        monotonic_rate,
        1.0,
        "그룹별 시계열 증가 순서 일관성 비율",
    )

    null_share_threshold = float(criteria.get("max_null_share_in_latest_step", 0.2))
    if latest_step_null_share is None:
        latest_null_status = "WARN"
    elif float(latest_step_null_share) <= null_share_threshold:
        latest_null_status = "PASS"
    else:
        latest_null_status = "WARN"
    _add_check(
        checks,
        "avh_latest_step_null_share",
        "WARN",
        latest_null_status,
        latest_step_null_share,
        null_share_threshold,
        "최신 스텝 피처 결측 비율",
    )

    shift_threshold = float(criteria.get("max_numeric_shift_score", 4.0))
    if numeric_shift_score is None:
        shift_status = "WARN"
    elif float(numeric_shift_score) <= shift_threshold:
        shift_status = "PASS"
    else:
        shift_status = "FAIL"
    _add_check(
        checks,
        "avh_numeric_shift_score",
        "HARD",
        shift_status,
        numeric_shift_score,
        shift_threshold,
        "최신 스텝 수치 변화 강도(robust score)",
    )

    target_flip_threshold = float(criteria.get("max_target_flip_rate", 0.5))
    if target_flip_rate is None:
        target_flip_status = "WARN"
    elif float(target_flip_rate) <= target_flip_threshold:
        target_flip_status = "PASS"
    else:
        target_flip_status = "WARN"
    _add_check(
        checks,
        "avh_target_flip_rate",
        "WARN",
        target_flip_status,
        target_flip_rate,
        target_flip_threshold,
        "최신 스텝 타깃 전환 비율",
    )

    error_message = avh_summary.get("error")
    if error_message:
        _add_check(
            checks,
            "avh_runtime_error",
            "HARD",
            "FAIL",
            error_message,
            None,
            "AVH 요약 생성 중 오류",
        )

    decision = _decide_overall(checks)
    return {
        "decision": decision,
        "checks": checks,
        "recommendation_text": _recommendation_text(decision),
    }


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

    task_type = dataset_meta.get("task_type")
    if task_type == "Auto Validate by History":
        return _evaluate_auto_validate_by_history(profile, policy)

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
