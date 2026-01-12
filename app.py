import json

import pandas as pd
import streamlit as st

from core.evaluator import evaluate_o1
from core.policy import load_policy
from core.profiler import build_profile_o1

st.set_page_config(page_title="데이터 검증 Agent", layout="wide")

st.title("데이터 검증 Agent")

POLICY_PATH_O1 = "policies/o1_training_default.yaml"

OBJECTIVES = [
    {"key": "model_training_readiness", "label": "모델 학습 데이터 품질"},
    {"key": "model_evaluation_readiness", "label": "모델 평가 데이터 품질"},
    {"key": "deployment_serving_readiness", "label": "배포/서빙 Readiness"},
    {"key": "agent_automation_readiness", "label": "Agent/Automation Readiness"},
]

OBJECTIVE_LABELS = {item["key"]: item["label"] for item in OBJECTIVES}

if "objective" not in st.session_state:
    st.session_state["objective"] = None

if "objective_settings" not in st.session_state:
    st.session_state["objective_settings"] = {}

if "profile_cache" not in st.session_state:
    st.session_state["profile_cache"] = {}

if "report_cache" not in st.session_state:
    st.session_state["report_cache"] = {}


def _get_columns() -> list[str]:
    df = st.session_state.get("uploaded_df")
    if df is None:
        return []
    return list(df.columns)


def _save_settings(objective_key: str, settings: dict) -> None:
    st.session_state["objective_settings"][objective_key] = settings
    st.session_state["profile_cache"] = {}
    st.session_state["report_cache"] = {}


def _build_profile_context(
    objective_key: str, settings: dict, df: pd.DataFrame
) -> dict:
    return {
        "objective_key": objective_key,
        "settings": settings,
        "shape": tuple(df.shape),
        "columns": list(df.columns),
    }


def _get_or_build_profile(
    objective_key: str, settings: dict, df: pd.DataFrame
) -> tuple[dict | None, str | None]:
    context = _build_profile_context(objective_key, settings, df)
    cached = st.session_state.get("profile_cache", {})
    if cached.get("context") == context:
        return cached.get("data"), cached.get("error")

    profile = None
    error = None
    try:
        profile = build_profile_o1(
            df,
            task_type=settings.get("task_type"),
            target_col=settings.get("target_column"),
            group_key=settings.get("group_key"),
            time_col=settings.get("time_column"),
            sampling_mode=settings.get("sampling_mode", "auto"),
            sample_n=settings.get("sample_n", 5000),
        )
    except Exception as exc:
        error = str(exc)

    st.session_state["profile_cache"] = {
        "context": context,
        "data": profile,
        "error": error,
    }
    return profile, error


def _get_or_build_report(
    profile_context: dict, profile: dict, policy: dict
) -> tuple[dict | None, str | None]:
    context = {"profile_context": profile_context, "policy": policy}
    cached = st.session_state.get("report_cache", {})
    if cached.get("context") == context:
        return cached.get("data"), cached.get("error")

    report = None
    error = None
    try:
        report = evaluate_o1(profile, policy)
    except Exception as exc:
        error = str(exc)

    st.session_state["report_cache"] = {
        "context": context,
        "data": report,
        "error": error,
    }
    return report, error


def render_objective_selector() -> None:
    st.subheader("목표를 선택하세요")

    cols = st.columns(4)
    for col, option in zip(cols, OBJECTIVES):
        with col:
            with st.container(border=True):
                st.markdown(f"### {option['label']}")
                st.write(" ")
                if st.button("선택", key=f"select_{option['key']}"):
                    st.session_state["objective"] = option["key"]
                    st.rerun()


def render_current_settings(objective_key: str) -> None:
    st.subheader("현재 설정")
    current_settings = st.session_state["objective_settings"].get(objective_key)
    if current_settings:
        st.json(current_settings)
    else:
        st.info("아직 저장된 설정이 없습니다.")


def render_csv_section(
    objective_key: str, settings: dict | None, policy: dict | None
) -> None:
    st.divider()
    uploaded = st.file_uploader("CSV 파일 업로드", type=["csv"])
    if uploaded is None:
        st.info("계속하려면 CSV 파일을 업로드하세요.")
        return

    try:
        df = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"CSV를 읽는 데 실패했습니다: {exc}")
        return

    st.session_state["uploaded_df"] = df
    st.caption(f"행: {len(df)} | 열: {len(df.columns)}")

    profile = None
    profile_error = None
    profile_context = None
    if objective_key == "model_training_readiness" and settings:
        profile_context = _build_profile_context(objective_key, settings, df)
        profile, profile_error = _get_or_build_profile(objective_key, settings, df)

    preview_tab, profile_tab, evaluation_tab, export_tab = st.tabs(
        ["미리보기", "프로파일", "평가", "내보내기"]
    )
    with preview_tab:
        st.dataframe(df.head(50), use_container_width=True)
    with profile_tab:
        if objective_key == "model_training_readiness":
            if not settings:
                st.info("프로파일을 생성하려면 사이드바에서 설정을 저장하세요.")
            elif profile_error:
                st.error(f"프로파일 생성에 실패했습니다: {profile_error}")
            else:
                st.json(profile)
        else:
            st.info("프로파일 화면은 준비 중입니다.")
    with evaluation_tab:
        if objective_key == "model_training_readiness":
            if not settings:
                st.info("평가를 진행하려면 사이드바 설정을 저장하세요.")
            elif profile_error:
                st.error(f"프로파일 생성에 실패했습니다: {profile_error}")
            elif profile is None:
                st.info("프로파일을 생성할 수 없습니다.")
            elif not policy:
                st.info("정책을 로드할 수 없습니다.")
            else:
                report, report_error = _get_or_build_report(
                    profile_context, profile, policy
                )
                if report_error:
                    st.error(f"평가에 실패했습니다: {report_error}")
                else:
                    st.metric("결정", report.get("decision", "UNKNOWN"))
                    st.write(report.get("recommendation_text", ""))
                    checks = report.get("checks", [])
                    if checks:
                        st.dataframe(pd.DataFrame(checks), use_container_width=True)
        else:
            st.info("평가 화면은 준비 중입니다.")
    with export_tab:
        if objective_key == "model_training_readiness":
            if profile:
                profile_json = json.dumps(profile, ensure_ascii=False, indent=2)
                st.download_button(
                    "프로파일 다운로드",
                    data=profile_json,
                    file_name="profile.json",
                    mime="application/json",
                )
            else:
                st.info("프로파일을 생성해야 다운로드할 수 있습니다.")

            if policy and profile and profile_context:
                report, report_error = _get_or_build_report(
                    profile_context, profile, policy
                )
                if report_error:
                    st.error(f"평가에 실패했습니다: {report_error}")
                elif report:
                    report_json = json.dumps(report, ensure_ascii=False, indent=2)
                    st.download_button(
                        "평가 리포트 다운로드",
                        data=report_json,
                        file_name="evaluation_report.json",
                        mime="application/json",
                    )
            else:
                st.info("평가 리포트 생성을 위해 프로파일과 정책이 필요합니다.")
        else:
            st.info("내보내기 옵션은 준비 중입니다.")


def render_training_sidebar() -> None:
    columns = _get_columns()
    with st.sidebar.form("training_readiness_form"):
        task_type = st.selectbox("작업 유형", ["classification", "regression"])

        if columns:
            target_column = st.selectbox("타깃 컬럼", columns)
            group_key = st.selectbox("그룹 키(선택)", ["(없음)"] + columns)
            time_column = st.selectbox("시간 컬럼(선택)", ["(없음)"] + columns)
        else:
            target_column = st.text_input("타깃 컬럼")
            group_key = st.text_input("그룹 키(선택)")
            time_column = st.text_input("시간 컬럼(선택)")

        sampling_mode = st.selectbox("샘플링 모드", ["auto", "full", "sample"])
        sample_n = st.number_input("샘플 수", min_value=1, value=1000, step=100)
        submitted = st.form_submit_button("설정 저장")

    if submitted:
        resolved_group_key = None
        if group_key and group_key != "(없음)":
            resolved_group_key = group_key

        resolved_time_column = None
        if time_column and time_column != "(없음)":
            resolved_time_column = time_column

        _save_settings(
            "model_training_readiness",
            {
                "task_type": task_type,
                "target_column": target_column,
                "group_key": resolved_group_key,
                "time_column": resolved_time_column,
                "sampling_mode": sampling_mode,
                "sample_n": int(sample_n),
            },
        )


def render_placeholder_sidebar(objective_key: str) -> None:
    with st.sidebar.form(f"placeholder_form_{objective_key}"):
        st.info("아직 구현되지 않았습니다.")
        name = st.text_input("이름")
        notes = st.text_area("메모")
        submitted = st.form_submit_button("설정 저장")

    if submitted:
        _save_settings(
            objective_key,
            {"name": name, "notes": notes, "status": "not_implemented"},
        )


def run_o1_training(policy: dict | None) -> None:
    st.sidebar.header("설정")
    if policy:
        st.sidebar.subheader("정책 임계값")
        st.sidebar.json(policy)
    else:
        st.sidebar.error("정책을 로드할 수 없습니다.")
    render_training_sidebar()


def run_o2_evaluation() -> None:
    st.sidebar.header("설정")
    render_placeholder_sidebar("model_evaluation_readiness")


def run_o3_deployment() -> None:
    st.sidebar.header("설정")
    render_placeholder_sidebar("deployment_serving_readiness")


def run_o4_agent() -> None:
    st.sidebar.header("설정")
    render_placeholder_sidebar("agent_automation_readiness")


objective_key = st.session_state["objective"]

if objective_key:
    st.header(f"선택한 목표: {OBJECTIVE_LABELS.get(objective_key, objective_key)}")
    if st.button("목표 변경"):
        st.session_state["objective"] = None
        st.session_state.pop("uploaded_df", None)
        st.session_state["profile_cache"] = {}
        st.session_state["report_cache"] = {}
        st.rerun()

    render_current_settings(objective_key)
    current_settings = st.session_state["objective_settings"].get(objective_key)

    policy_o1 = None
    if objective_key == "model_training_readiness":
        try:
            policy_o1 = load_policy(POLICY_PATH_O1)
        except Exception:
            policy_o1 = None

    render_csv_section(objective_key, current_settings, policy_o1)

    if objective_key == "model_training_readiness":
        run_o1_training(policy_o1)
    elif objective_key == "model_evaluation_readiness":
        run_o2_evaluation()
    elif objective_key == "deployment_serving_readiness":
        run_o3_deployment()
    elif objective_key == "agent_automation_readiness":
        run_o4_agent()
else:
    render_objective_selector()
