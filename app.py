import io
import json
import os
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from core.evaluator import evaluate_o1
from core.llm_client import call_ollama_generate
from core.policy import load_policy
from core.profiler import build_profile_o1
try:
    from auto_validate_by_history.streamlit_runner import (
        get_avh_evaluation_criteria,
        run_auto_validate_by_history,
    )
except Exception:
    get_avh_evaluation_criteria = None
    run_auto_validate_by_history = None
try:
    from auto_validate_by_history.workflow import (
        build_rule_candidates,
        evaluate_ground_truth,
        generate_synthetic_anomalies,
        prepare_history,
        select_rules_with_budget,
    )
except Exception:
    build_rule_candidates = None
    evaluate_ground_truth = None
    generate_synthetic_anomalies = None
    prepare_history = None
    select_rules_with_budget = None

st.set_page_config(
    page_title="데이터 품질 Agent",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("데이터 품질 Agent")

def _inject_global_font_css() -> None:
    st.markdown(
        """
<style>
html, body, [class*="css"], [data-testid="stAppViewContainer"], [data-testid="stSidebar"] {
  font-family: "Noto Sans KR", "Malgun Gothic", "Apple SD Gothic Neo", "Segoe UI", sans-serif !important;
}
</style>
""",
        unsafe_allow_html=True,
    )


_inject_global_font_css()

POLICY_PATH_O1 = "policies/o1_training_default.yaml"

SCHEMA_BLOCK_GENERAL_DB = """
You are helping with analysis on a telco customer relational database.

We have the following tables and columns:

1) customers
- "Customer ID" (string, primary key)  -> customer_id
- Age (int)
- Gender (string)
- State (string)
- City (string)
- "Zip Code" (string)
- "Tenure in Months" (int)
- Contract (string)                 -- current contract type
- "Payment Method" (string)         -- current payment method
- "Avg Monthly GB Download" (float)
- "Avg Monthly Long Distance Charges" (float)
- "Satisfaction Score" (int)
- CLTV (float)
- Churn (int, 1=churned, 0=active)
- "Customer Status" (string)

2) churn_history_v2
- event_id (string, primary key)
- customer_id (string)
- event_seq (int, 1~N, time order)
- event_month (date)
- status (string: JOINED/STAYED/CHURNED/REJOINED)
- churn_flag (int, 1 if CHURNED, else 0)
- churn_category (string, nullable)
- churn_reason (string, nullable)
- churn_score (float, nullable)

3) internet_service_gaip_history_v3
- contact_id (string)
- customer_id (string)
- event_seq (int)
- event_month (date)
- "Internet Service"       (int or bool)
- "Internet Type"          (string)
- "Online Backup"          (int or bool)
- "Online Security"        (int or bool)
- "Device Protection Plan" (int or bool)
- "Premium Tech Support"   (int or bool)
- "Phone Service"          (int or bool)
- "Multiple Lines"         (int or bool)
- "Streaming Movies"       (int or bool)
- "Streaming Music"        (int or bool)
- "Streaming TV"           (int or bool)
- "Unlimited Data"         (int or bool)

4) usage_history_v1
- contact_id (string)
- customer_id (string)
- event_seq (int)
- event_month (date)
- Monthly_GB_Download (float)
- Monthly_Long_Distance_Charges (float)

5) contract_option_history_v1
- contact_id (string)
- customer_id (string)
- event_seq (int)
- event_month (date)
- Contract (string)
- Payment_Method (string)

6) billing_history_v1
- contact_id (string)
- customer_id (string)
- event_seq (int)
- event_month (date)
- Revenue (float)
- Charge (float)
- Extra_Data_Charges (float)
- Refunds (float)
- Offer (string, nullable)

Keys:
- customers."Customer ID" = all history tables' customer_id
- history tables can be joined with (customer_id, event_seq)
""".strip()

SCHEMA_BLOCK_ONTOLOGY_CONTEXT = """
You are helping with analysis on a telco customer ontology-like graph.

You will be given:
- [Ontology Nodes] table: each row is a node with:
  - node_id, node_type, customer_id, event_month, properties(JSON)
- [Ontology Edges] table: each row is a directed relationship with:
  - src_id, rel_type, dst_id, properties(JSON)

Node types:
- Customer: static customer profile node
- CustomerMonth: monthly snapshot node (time slice)
- UsageRecord: monthly usage fact node
- BillingRecord: monthly billing fact node
- SubscriptionState: monthly subscription/options state node

Relationship types:
- HAS_MONTH_STATE: Customer -> CustomerMonth
- HAS_USAGE: CustomerMonth -> UsageRecord
- HAS_BILLING: CustomerMonth -> BillingRecord
- HAS_SUBSCRIPTION_STATE: CustomerMonth -> SubscriptionState
- NEXT: CustomerMonth -> CustomerMonth (next month)

Keys/notes:
- You must answer based ONLY on the provided Ontology Context tables.
- event_month is the time key. Consider time order when explaining changes.
""".strip()

OBJECTIVES = [
    {"key": "model_training_readiness", "label": "데이터 학습 품질"},
    {"key": "agent_automation_readiness", "label": "온톨로지 DB 품질"},
]

OBJECTIVE_LABELS = {item["key"]: item["label"] for item in OBJECTIVES}

AGENT_POC_FEATURES = [
    "온톨로지 데이터 평가 PoC",
]
POC_PAGES = {
    "온톨로지 데이터 평가 PoC": "ontology_data_eval",
}

if "objective" not in st.session_state:
    st.session_state["objective"] = None

if "objective_settings" not in st.session_state:
    st.session_state["objective_settings"] = {}

if "profile_cache" not in st.session_state:
    st.session_state["profile_cache"] = {}

if "report_cache" not in st.session_state:
    st.session_state["report_cache"] = {}


def _is_cloud_runtime() -> bool:
    return os.environ.get("STREAMLIT_RUNTIME_ENV") == "cloud" or os.environ.get(
        "STREAMLIT_CLOUD"
    ) == "true"


def _get_columns() -> list[str]:
    df = st.session_state.get("uploaded_df")
    if df is None:
        return []
    return list(df.columns)


def _to_streamlit_safe_df(df: pd.DataFrame) -> pd.DataFrame:
    safe_df = df.copy()
    for col in safe_df.columns:
        series = safe_df[col]
        if series.dtype != "object":
            continue
        non_null = series.dropna()
        if non_null.empty:
            continue
        value_types = {type(value) for value in non_null}
        if len(value_types) > 1:
            safe_df[col] = series.map(lambda value: value if pd.isna(value) else str(value))
    return safe_df


def _highlight_current_value(df: pd.DataFrame):
    if "current_value" not in df.columns:
        return df
    return df.style.set_properties(subset=["current_value"], **{"font-weight": "700"})


def _build_distance_signal_summary(distance_df: pd.DataFrame) -> pd.DataFrame:
    if distance_df.empty or "feature" not in distance_df.columns or "z_score" not in distance_df.columns:
        return pd.DataFrame()
    working = distance_df.copy()
    working["abs_z_score"] = pd.to_numeric(working["z_score"], errors="coerce").abs()
    working["current_value"] = pd.to_numeric(working["current_value"], errors="coerce")
    working = working.dropna(subset=["abs_z_score"])
    if working.empty:
        return pd.DataFrame()
    top = (
        working.sort_values(["feature", "abs_z_score"], ascending=[True, False])
        .groupby("feature", as_index=False)
        .first()
    )
    return top[["feature", "metric", "current_value", "hist_mean", "hist_std", "z_score", "abs_z_score"]]


def _save_settings(objective_key: str, settings: dict) -> None:
    st.session_state["objective_settings"][objective_key] = settings
    st.session_state["profile_cache"] = {}
    st.session_state["report_cache"] = {}
    for key in (
        "avh_pipeline",
        "avh_synthetic_df",
        "avh_rule_bundle",
        "avh_eval_result",
        "avh_selected_rules_df",
    ):
        st.session_state.pop(key, None)


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


def _get_or_build_avh_profile(
    objective_key: str, settings: dict, df: pd.DataFrame
) -> tuple[dict | None, str | None]:
    context = _build_profile_context(objective_key, settings, df)
    cached = st.session_state.get("profile_cache", {})
    if cached.get("context") == context and cached.get("mode") == "avh_direct":
        return cached.get("data"), cached.get("error")

    profile = None
    error = None
    if run_auto_validate_by_history is None:
        error = (
            "auto_validate_by_history.streamlit_runner.run_auto_validate_by_history "
            "를 로드할 수 없습니다."
        )
    else:
        try:
            avh_summary = run_auto_validate_by_history(
                df=df,
                group_key=settings.get("group_key"),
                time_col=settings.get("time_column"),
                target_col=settings.get("target_column"),
            )
            profile = {
                "dataset_meta": {
                    "task_type": "Auto Validate by History",
                    "total_rows": int(len(df)),
                    "sampled_rows": int(len(df)),
                    "sampled": False,
                    "target_column": settings.get("target_column"),
                    "group_key": settings.get("group_key"),
                    "time_column": settings.get("time_column"),
                },
                "schema": [
                    {"name": str(col), "dtype": str(dtype)}
                    for col, dtype in df.dtypes.items()
                ],
                "columns_summary": {},
                "target_summary": {"avh_summary": avh_summary},
            }
        except Exception as exc:
            error = str(exc)

    st.session_state["profile_cache"] = {
        "context": context,
        "mode": "avh_direct",
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


def _df_to_markdown(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except Exception:
        return df.to_csv(index=False)


def _read_csv_with_fallback(
    uploaded_file,
) -> tuple[pd.DataFrame | None, str | None, str | None]:
    raw = uploaded_file.getvalue()
    encodings = ("utf-8-sig", "utf-8", "cp949", "euc-kr")
    last_error = None
    for encoding in encodings:
        try:
            df = pd.read_csv(io.BytesIO(raw), encoding=encoding)
            return df, encoding, None
        except Exception as exc:
            last_error = str(exc)
    return None, None, last_error


def _decode_text_with_fallback(
    uploaded_file,
) -> tuple[str | None, str | None, str | None]:
    raw = uploaded_file.getvalue()
    encodings = ("utf-8-sig", "utf-8", "cp949", "euc-kr")
    last_error = None
    for encoding in encodings:
        try:
            return raw.decode(encoding), encoding, None
        except Exception as exc:
            last_error = str(exc)
    return None, None, last_error


def _load_properties(value: object) -> dict:
    if isinstance(value, dict):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return {}
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return {}
    return {}


def _extract_properties(df: pd.DataFrame) -> pd.DataFrame:
    if "properties" not in df.columns:
        return pd.DataFrame(index=df.index)
    props = df["properties"].apply(_load_properties)
    props_df = pd.json_normalize(props)
    props_df.index = df.index
    return props_df


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lowered = {col.lower(): col for col in df.columns}
    for name in candidates:
        if name in df.columns:
            return name
        if name.lower() in lowered:
            return lowered[name.lower()]
    return None


def _filter_by_customer_id(
    df: pd.DataFrame, customer_id: str | None
) -> pd.DataFrame:
    if df.empty or not customer_id:
        return df
    col = _find_column(df, ["customer_id", "customer id", "Customer ID"])
    if not col:
        return df
    return df[df[col].astype(str) == str(customer_id)]


def _filter_edges_by_nodes(
    edges_df: pd.DataFrame, nodes_df: pd.DataFrame
) -> pd.DataFrame:
    if edges_df.empty or nodes_df.empty:
        return edges_df
    if "node_id" not in nodes_df.columns:
        return edges_df
    if "src_id" not in edges_df.columns or "dst_id" not in edges_df.columns:
        return edges_df
    node_ids = set(nodes_df["node_id"].astype(str))
    return edges_df[
        edges_df["src_id"].astype(str).isin(node_ids)
        | edges_df["dst_id"].astype(str).isin(node_ids)
    ]


def _build_ontology_context_markdown(
    nodes_df: pd.DataFrame, edges_df: pd.DataFrame, max_rows_each: int = 200
) -> str:
    nodes_view = nodes_df.copy()
    edges_view = edges_df.copy()
    if "properties" in nodes_view.columns:
        nodes_view["properties"] = (
            nodes_view["properties"].astype(str).str.slice(0, 350)
        )
    if "properties" in edges_view.columns:
        edges_view["properties"] = (
            edges_view["properties"].astype(str).str.slice(0, 350)
        )

    nodes_md = _df_to_markdown(nodes_view.head(max_rows_each))
    edges_md = _df_to_markdown(edges_view.head(max_rows_each))
    return (
        "[Ontology Nodes]\n"
        f"{nodes_md}\n\n"
        "[Ontology Edges]\n"
        f"{edges_md}"
    )


def _build_ontology_timeline_from_nodes(
    nodes_df: pd.DataFrame | None, customer_id: str | None
) -> pd.DataFrame:
    if nodes_df is None or nodes_df.empty:
        return pd.DataFrame()
    working = _filter_by_customer_id(nodes_df, customer_id)
    if "node_type" not in working.columns:
        return pd.DataFrame()

    def _base_frame(node_type: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        subset = working[working["node_type"] == node_type].copy()
        if "customer_id" not in subset.columns:
            subset["customer_id"] = customer_id if customer_id else None
        if "event_month" not in subset.columns:
            subset["event_month"] = None
        if subset.empty:
            return subset, pd.DataFrame(index=subset.index)
        props = _extract_properties(subset)
        subset["event_month"] = props.get(
            "event_month", subset.get("event_month")
        )
        return subset, props

    cm, cm_props = _base_frame("CustomerMonth")
    if cm.empty:
        return pd.DataFrame()

    cm_out = pd.DataFrame(
        {
            "customer_id": cm["customer_id"],
            "event_seq": cm_props.get("event_seq", cm.get("event_seq")),
            "event_month": cm["event_month"],
            "status": cm_props.get("status"),
            "churn_flag": cm_props.get("churn_flag"),
            "churn_category": cm_props.get("churn_category"),
            "churn_reason": cm_props.get("churn_reason"),
            "churn_score": cm_props.get("churn_score"),
        }
    )

    u, u_props = _base_frame("UsageRecord")
    u_out = pd.DataFrame(
        {
            "customer_id": u["customer_id"],
            "event_month": u["event_month"],
            "Monthly_GB_Download": u_props.get("Monthly_GB_Download"),
            "Monthly_Long_Distance_Charges": u_props.get(
                "Monthly_Long_Distance_Charges"
            ),
        }
    )

    b, b_props = _base_frame("BillingRecord")
    b_out = pd.DataFrame(
        {
            "customer_id": b["customer_id"],
            "event_month": b["event_month"],
            "Revenue": b_props.get("Revenue"),
            "Charge": b_props.get("Charge"),
            "Extra_Data_Charges": b_props.get("Extra_Data_Charges"),
            "Refunds": b_props.get("Refunds"),
            "Offer": b_props.get("Offer"),
        }
    )

    s, s_props = _base_frame("SubscriptionState")
    s_out = pd.DataFrame(
        {
            "customer_id": s["customer_id"],
            "event_month": s["event_month"],
            "Internet_Service": s_props.get("Internet_Service"),
            "Internet_Type": s_props.get("Internet_Type"),
            "Online_Backup": s_props.get("Online_Backup"),
            "Online_Security": s_props.get("Online_Security"),
            "Device_Protection_Plan": s_props.get("Device_Protection_Plan"),
            "Premium_Tech_Support": s_props.get("Premium_Tech_Support"),
            "Phone_Service": s_props.get("Phone_Service"),
            "Multiple_Lines": s_props.get("Multiple_Lines"),
            "Streaming_Movies": s_props.get("Streaming_Movies"),
            "Streaming_Music": s_props.get("Streaming_Music"),
            "Streaming_TV": s_props.get("Streaming_TV"),
            "Unlimited_Data": s_props.get("Unlimited_Data"),
            "Contract": s_props.get("Contract"),
            "Payment_Method": s_props.get("Payment_Method"),
        }
    )

    timeline = cm_out.merge(
        u_out, on=["customer_id", "event_month"], how="left"
    )
    timeline = timeline.merge(
        b_out, on=["customer_id", "event_month"], how="left"
    )
    timeline = timeline.merge(
        s_out, on=["customer_id", "event_month"], how="left"
    )
    return timeline


def _build_general_db_prompt(
    question: str, customer_id: str, timeline_df: pd.DataFrame
) -> str:
    if timeline_df.empty:
        table_md = "(no timeline data for the requested customer_id)"
    else:
        table_md = _df_to_markdown(timeline_df)
    prompt = f"""
{SCHEMA_BLOCK_GENERAL_DB}

Below is a unified timeline for customer_id = "{customer_id}".
Each row uses event_seq (1 is oldest, larger numbers are more recent).

Data table:
{table_md}

Question:
{question}

Requirements:
- Answer in Korean.
- Explain only using values from the table; do not over-speculate.
- Use time order (event_seq) and describe usage/charges/service/status changes (including churn/rejoin).
- Mention important numeric changes (e.g., Monthly_GB_Download, Revenue) with values when possible.
"""
    return prompt.strip()


def _build_ontology_prompt(
    question: str,
    customer_id: str,
    ontology_timeline_df: pd.DataFrame,
    ontology_context: str,
) -> str:
    if ontology_timeline_df.empty:
        table_md = "(no ontology-based monthly timeline)"
    else:
        table_md = _df_to_markdown(ontology_timeline_df)
    prompt = f"""
{SCHEMA_BLOCK_ONTOLOGY_CONTEXT}

Below is the Ontology Context for customer_id = "{customer_id}".
- Nodes/Edges include monthly state/usage/billing/service-option changes.
- Consider time order using event_month or NEXT edges.

[Ontology-derived Monthly Timeline]
{table_md}

Data tables:
{ontology_context}

Question:
{question}

Requirements:
- Answer in Korean.
- Explain only using values from the tables; do not over-speculate.
- Use time order (event_seq or event_month) and describe usage/charges/service/status changes (including churn/rejoin).
- Mention important numeric changes (e.g., Monthly_GB_Download, Revenue) with values when possible.
"""
    return prompt.strip()


def _render_avh_profile(profile: dict) -> None:
    target_summary = profile.get("target_summary", {})
    avh = target_summary.get("avh_summary", {}) if isinstance(target_summary, dict) else {}
    if not avh:
        st.info("Auto Validate by History 결과가 없습니다.")
        return

    st.caption(f"engine: {avh.get('engine', '-')}")
    if avh.get("error"):
        st.error(f"AVH 실행 오류: {avh.get('error')}")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Rows", avh.get("total_rows", "-"))
    c2.metric("Total Groups", avh.get("total_groups", "-"))
    c3.metric("Usable Groups", avh.get("usable_history_groups", "-"))
    c4.metric("History Ready Rate", avh.get("history_ready_rate", "-"))

    c5, c6, c7 = st.columns(3)
    c5.metric("Avg History Length", avh.get("avg_history_length", "-"))
    c6.metric("Median History Length", avh.get("median_history_length", "-"))
    c7.metric("Monotonic Rate", avh.get("sequence_monotonic_rate", "-"))

    c8, c9, c10, c11 = st.columns(4)
    c8.metric("Datetime Parse Success", avh.get("datetime_parse_success_rate", "-"))
    c9.metric("Group Size CV", avh.get("group_size_cv", "-"))
    c10.metric("Dup (Group,Time) Rate", avh.get("duplicate_group_time_rate", "-"))
    c11.metric("Non-monotonic Group Rate", avh.get("non_monotonic_group_rate", "-"))

    c12, c13, c14 = st.columns(3)
    c12.metric("Latest Step Null Share", avh.get("latest_step_null_share", "-"))
    c13.metric("Numeric Shift Score", avh.get("numeric_shift_score", "-"))
    c14.metric("Latest Step Groups", avh.get("latest_step_group_count", "-"))

    top_delta = avh.get("top_numeric_delta") or {}
    if top_delta:
        st.markdown("**Top Numeric Delta**")
        st.dataframe(
            pd.DataFrame(
                [{"column": k, "avg_abs_delta": v} for k, v in top_delta.items()]
            ),
            use_container_width=True,
        )

    shift_detail = avh.get("numeric_shift_by_feature") or {}
    if shift_detail:
        st.markdown("**Numeric Shift By Feature**")
        st.dataframe(
            pd.DataFrame(
                [{"column": k, "shift_score": v} for k, v in shift_detail.items()]
            ).sort_values("shift_score", ascending=False),
            use_container_width=True,
        )

    if avh.get("target_last_step_change_rate") is not None:
        st.metric(
            "Target Last-step Change Rate",
            avh.get("target_last_step_change_rate"),
        )

    with st.expander("AVH Raw JSON"):
        st.json(avh)


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


def render_avh_pipeline_section(settings: dict | None) -> None:
    st.divider()
    st.subheader("Auto Validate by History")
    st.caption(
        "1) 이력 업로드(30일+) -> 2) Feature 벡터 -> 3) Synthetic 이상치 -> "
        "4) Budget 기반 규칙 조합 -> 5) 정답 업로드 -> 6) Precision/Recall -> 7) 시각화"
    )

    if prepare_history is None:
        st.error("auto_validate_by_history.workflow 모듈을 로드할 수 없습니다.")
        return

    history_file = st.file_uploader(
        "1) 이력 데이터 CSV 업로드 (동일 포맷, 최소 30일 이상)",
        type=["csv"],
        key="avh_history_csv",
    )
    if history_file is None:
        st.info("이력 데이터를 업로드하면 단계별 평가를 진행할 수 있습니다.")
        return

    history_df, detected_encoding, read_error = _read_csv_with_fallback(history_file)
    if history_df is None:
        st.error(f"CSV를 읽는 데 실패했습니다: {read_error}")
        return
    if detected_encoding and detected_encoding not in ("utf-8-sig", "utf-8"):
        st.info(f"CSV 인코딩을 `{detected_encoding}`로 감지해 로드했습니다.")
    st.caption(f"이력 데이터: 행 {len(history_df)} | 열 {len(history_df.columns)}")
    with st.expander("이력 데이터 미리보기 (상위 100행)"):
        st.dataframe(history_df.head(100), use_container_width=True)

    cols = list(history_df.columns)
    default_group = settings.get("group_key") if settings else None
    default_time = settings.get("time_column") if settings else None
    group_idx = cols.index(default_group) if default_group in cols else 0
    time_idx = cols.index(default_time) if default_time in cols else 0

    c1, c2 = st.columns(2)
    with c1:
        selected_group = st.selectbox("그룹 키 컬럼", cols, index=group_idx, key="avh_group_col")
    with c2:
        selected_time = st.selectbox("시간 컬럼", cols, index=time_idx, key="avh_time_col")

    if st.button("2) Feature 벡터 계산", key="avh_prepare_btn"):
        prepared = prepare_history(history_df, selected_group, selected_time)
        st.session_state["avh_pipeline"] = prepared
        st.session_state.pop("avh_synthetic_df", None)
        st.session_state.pop("avh_rule_bundle", None)
        st.session_state.pop("avh_eval_result", None)
        st.session_state.pop("avh_selected_rules_df", None)

    prepared = st.session_state.get("avh_pipeline")
    if not prepared:
        return
    if prepared.get("error"):
        st.error(prepared["error"])
        return

    history_check = prepared.get("history_check", {})
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Unique Days", history_check.get("unique_days", "-"))
    m2.metric("Required Days", history_check.get("required_min_days", 30))
    m3.metric("History Ready", "YES" if history_check.get("history_ready") else "NO")
    m4.metric("Datetime Parse Rate", history_check.get("parse_rate", "-"))
    st.markdown(
        """
**지표 설명**

- `Unique Days`: 이력 데이터의 고유 일자 수
- `Required Days`: 최소 필요 일수
- `History Ready`: 최소 이력 요건 충족 여부
- `Datetime Parse Rate`: 시간 컬럼의 날짜 변환 성공 비율
"""
    )
    if not history_check.get("history_ready"):
        st.error("최소 30일 요건을 만족하지 않아 다음 단계(3~7)를 진행할 수 없습니다.")
        return

    st.markdown("**2-1) Feature 벡터(통계)**")
    feature_df = prepared.get("feature_df", pd.DataFrame())
    if feature_df.empty:
        st.warning("Feature 벡터를 계산할 수 없습니다.")
        return
    st.dataframe(feature_df, use_container_width=True)
    st.markdown(
        """
**항목 설명**

- `mean`: 전체 이력 평균
- `std`: 전체 이력 표준편차
- `median`: 전체 이력 중앙값
- `mad`: 중앙값 기준 절대편차의 중앙값. 이상치에 덜 민감한 변동성 지표
- `p05` / `p95`: 하위 5%, 상위 95% 분위수
- `latest_mean` / `latest_std`: 각 그룹 최신 시점 값의 평균 / 표준편차
- `latest_prev_delta_mean`: 각 그룹의 최신값과 직전값 차이의 평균

**선정 기준**

- `group_key`, `time_col`을 제외한 수치형 컬럼만 대상
- 평소 분포, 최신 상태, 최근 변화까지 함께 보기 위한 구성
"""
    )

    st.markdown("**2-2) t vs t-1 거리 Feature 벡터**")
    distance_df = prepared.get("distance_df", pd.DataFrame())
    if distance_df.empty:
        st.info("거리 벡터를 계산할 수 없습니다.")
        distance_debug = prepared.get("distance_debug", {})
        if distance_debug:
            st.markdown("**확인 포인트**")
            d1, d2, d3 = st.columns(3)
            d1.metric("Total Groups", distance_debug.get("total_groups", "-"))
            d2.metric("Common t/t-1 Groups", distance_debug.get("common_group_count", "-"))
            d3.metric("Eligible Features", distance_debug.get("eligible_feature_count", "-"))
            per_feature = distance_debug.get("per_feature_valid_group_count", {})
            if per_feature:
                debug_df = pd.DataFrame(
                    [
                        {"feature": feature, "valid_group_count": count}
                        for feature, count in per_feature.items()
                    ]
                ).sort_values(["valid_group_count", "feature"], ascending=[False, True])
                st.markdown("각 수치형 컬럼이 t와 t-1 값을 모두 가진 그룹 수")
                st.dataframe(debug_df, use_container_width=True)
    else:
        st.dataframe(_highlight_current_value(distance_df.head(60)), use_container_width=True)
        st.markdown(
            """
**항목 설명**

- 각 행은 `feature x metric` 조합입니다.
- `current_value`는 가장 최근 day pair의 `t vs t-1` metric 값입니다.
- `hist_mean`, `hist_std`는 그 이전 day pair들에서 같은 metric의 히스토리 평균 / 표준편차입니다.
- `z_score`는 현재 값이 과거 히스토리 대비 얼마나 벗어났는지 보여줍니다.

**주요 metric**

- `EMD`, `JS_div`, `KL_div`, `KS_dist`, `Cohen_dist`: t와 t-1 분포 차이
- `Min`, `Max`, `Mean`, `Median`, `Count`, `Sum`, `Range`: latest(t) 쪽 단일 통계
- `Skew`, `2-moment`, `3-moment`, `unique_ratio`, `complete_ratio`: latest(t) 분포 형태 요약
"""
        )
        distance_history_df = prepared.get("distance_history_df", pd.DataFrame())
        if isinstance(distance_history_df, pd.DataFrame) and not distance_history_df.empty:
            with st.expander("최근 t vs t-1 거리 시계열 보기"):
                st.dataframe(distance_history_df.tail(60), use_container_width=True)

    if st.button("3) Synthetic 이상치 생성", key="avh_synthetic_btn"):
        syn_df = generate_synthetic_anomalies(prepared)
        st.session_state["avh_synthetic_df"] = syn_df
        st.session_state.pop("avh_rule_bundle", None)
        st.session_state.pop("avh_eval_result", None)
        st.session_state.pop("avh_selected_rules_df", None)

    synthetic_df = st.session_state.get("avh_synthetic_df")
    if synthetic_df is not None:
        if synthetic_df.empty:
            st.warning("Synthetic 이상치가 생성되지 않았습니다.")
            return
        st.markdown("**3) 생성된 Synthetic 이상치 샘플**")
        st.caption("2-1의 baseline center/scale을 사용해 spike, drop, missing, mean_shift 유형의 synthetic anomaly를 생성합니다.")
        st.caption(
            f"행 {len(synthetic_df)} | 이상치 비율: "
            f"{round(float(synthetic_df['__is_anomaly'].mean()), 4) if len(synthetic_df) else 0.0}"
        )
        s1, s2, s3 = st.columns(3)
        s1.metric("Synthetic Rows", int(len(synthetic_df)))
        s2.metric("Anomaly Rows", int(synthetic_df["__is_anomaly"].sum()))
        s3.metric("Normal Rows", int((synthetic_df["__is_anomaly"] == 0).sum()))
        preview_cols = [
            c
            for c in ("__is_anomaly", "__anomaly_type", "__anomaly_feature")
            if c in synthetic_df.columns
        ]
        ordered_cols = preview_cols + [c for c in synthetic_df.columns if c not in preview_cols]
        st.dataframe(synthetic_df[ordered_cols].head(80), use_container_width=True)
        if "__anomaly_type" in synthetic_df.columns:
            type_counts = (
                synthetic_df["__anomaly_type"].value_counts(dropna=False).rename_axis("anomaly_type").to_frame("count")
            )
            st.markdown("**3) Synthetic 유형 분포**")
            st.dataframe(type_counts, use_container_width=True)

        budget = st.number_input(
            "4) Budget (작을수록 보수적, 클수록 공격적)",
            min_value=0.0001,
            max_value=0.5,
            value=0.05,
            step=0.01,
            format="%.4f",
            key="avh_budget",
        )
        if st.button("4) Budget 기반 규칙 조합 생성", key="avh_rule_btn"):
            candidates = build_rule_candidates(prepared, synthetic_df)
            bundle = select_rules_with_budget(
                candidates=candidates,
                y_true=synthetic_df["__is_anomaly"].astype(int),
                budget=float(budget),
            )
            st.session_state["avh_rule_bundle"] = bundle
            st.session_state["avh_selected_rules_df"] = bundle.get("selected_df", pd.DataFrame())
            st.session_state.pop("avh_eval_result", None)

    rule_bundle = st.session_state.get("avh_rule_bundle")
    if rule_bundle:
        st.markdown("**4) 선택된 적합 규칙 조합**")
        selected_df = rule_bundle.get("selected_df", pd.DataFrame())
        if selected_df.empty:
            st.warning("현재 budget에서 선택 가능한 규칙이 없습니다. budget을 조금 키워보세요.")
        else:
            st.caption("선택된 규칙의 항목/임계치(threshold)는 좌측 사이드바의 Selected AVH Rules 섹션에 표시됩니다.")
            st.dataframe(_to_streamlit_safe_df(selected_df), use_container_width=True)
        metrics = rule_bundle.get("metrics", {})
        if metrics:
            r1, r2, r3 = st.columns(3)
            r1.metric("Synthetic Precision", metrics.get("precision", "-"))
            r2.metric("Synthetic Recall", metrics.get("recall", "-"))
            r3.metric("Synthetic FPR", metrics.get("fpr", "-"))
            st.caption("현재 규칙 조합이 synthetic anomaly를 얼마나 잘 잡는지 보는 중간 평가입니다.")

        distance_signal_df = _build_distance_signal_summary(prepared.get("distance_df", pd.DataFrame()))
        if not selected_df.empty and not distance_signal_df.empty:
            selected_view = selected_df.copy()
            if "feature" not in selected_view.columns and "column" in selected_view.columns:
                selected_view = selected_view.rename(columns={"column": "feature"})
            selected_view = selected_view.merge(distance_signal_df, on="feature", how="left")
            selected_view = selected_view.rename(
                columns={
                    "metric": "top_distance_metric",
                    "current_value": "distance_current",
                    "hist_mean": "distance_hist_mean",
                    "hist_std": "distance_hist_std",
                    "z_score": "distance_z_score",
                    "abs_z_score": "distance_abs_z_score",
                }
            )
            if "feature" in selected_df.columns:
                selected_view = selected_view.rename(columns={"feature": "column"})
            st.markdown("**4) 선택 규칙과 최근 거리 신호 연결**")
            st.caption("선택된 feature별로 2-2 단계의 가장 강한 히스토리 기반 거리 신호를 함께 보여줍니다.")
            st.dataframe(_to_streamlit_safe_df(selected_view), use_container_width=True)

        truth_file = st.file_uploader("5) 정답(라벨 포함) 데이터 CSV 업로드", type=["csv"], key="avh_truth_csv")
        if truth_file is not None:
            truth_df, truth_encoding, truth_error = _read_csv_with_fallback(truth_file)
            if truth_df is None:
                st.error(f"정답 CSV를 읽는 데 실패했습니다: {truth_error}")
                return
            if truth_encoding and truth_encoding not in ("utf-8-sig", "utf-8"):
                st.info(f"정답 CSV 인코딩을 `{truth_encoding}`로 감지해 로드했습니다.")
            st.caption(f"정답 데이터: 행 {len(truth_df)} | 열 {len(truth_df.columns)}")
            with st.expander("정답 데이터 미리보기 (상위 100행)"):
                st.dataframe(truth_df.head(100), use_container_width=True)
            label_candidates = [
                c
                for c in truth_df.columns
                if c.lower() in ("label", "is_anomaly", "anomaly_label", "target", "y")
            ]
            label_idx = truth_df.columns.get_loc(label_candidates[0]) if label_candidates else 0
            label_col = st.selectbox(
                "정답 라벨 컬럼(0/1)",
                list(truth_df.columns),
                index=label_idx,
                key="avh_label_col",
            )
            if st.button("6) 선택 규칙으로 정답 데이터 평가", key="avh_eval_btn"):
                eval_result = evaluate_ground_truth(
                    df=truth_df,
                    label_col=label_col,
                    selected_rules=rule_bundle.get("selected", []),
                    baseline_stats=prepared.get("baseline_stats", {}),
                )
                st.session_state["avh_eval_result"] = eval_result

    eval_result = st.session_state.get("avh_eval_result")
    if eval_result:
        if eval_result.get("error"):
            st.error(eval_result["error"])
            return
        st.markdown("**6) Precision / Recall 평가 결과**")
        metrics = eval_result.get("metrics", {})
        e1, e2, e3, e4 = st.columns(4)
        e1.metric("Precision", metrics.get("precision", "-"))
        e2.metric("Recall", metrics.get("recall", "-"))
        e3.metric("TP", metrics.get("tp", "-"))
        e4.metric("FP", metrics.get("fp", "-"))
        st.caption("5)에서 업로드한 정답 라벨 기준으로, 4)에서 선택한 규칙 조합의 성능을 평가한 결과입니다.")

        st.markdown("**7) 시각화**")
        cm_df = pd.DataFrame(
            {
                "item": ["TP", "FP", "TN", "FN"],
                "count": [
                    int(metrics.get("tp", 0) or 0),
                    int(metrics.get("fp", 0) or 0),
                    int(metrics.get("tn", 0) or 0),
                    int(metrics.get("fn", 0) or 0),
                ],
            }
        ).set_index("item")
        st.markdown("Confusion Matrix Count")
        st.bar_chart(cm_df)

        pr_df = pd.DataFrame(
            {
                "metric": ["precision", "recall"],
                "value": [
                    float(metrics.get("precision", 0.0) or 0.0),
                    float(metrics.get("recall", 0.0) or 0.0),
                ],
            }
        ).set_index("metric")
        st.markdown("Precision / Recall")
        st.bar_chart(pr_df)

        distance_signal_df = _build_distance_signal_summary(prepared.get("distance_df", pd.DataFrame()))
        if not distance_signal_df.empty:
            top_signal_df = (
                distance_signal_df.sort_values("abs_z_score", ascending=False)
                .head(10)[["feature", "abs_z_score"]]
                .set_index("feature")
            )
            st.markdown("Top Distance Signals")
            st.bar_chart(top_signal_df)


def render_csv_section(
    objective_key: str, settings: dict | None, policy: dict | None
) -> None:
    if (
        objective_key == "model_training_readiness"
        and settings
        and settings.get("task_type") == "Auto Validate by History"
    ):
        render_avh_pipeline_section(settings)
        return

    effective_policy = policy
    if settings and settings.get("task_type") == "Auto Validate by History":
        effective_policy = _load_avh_criteria()

    st.divider()
    if st.button("사내 데이터 불러오기 (Coming Soon)", key="internal_data_main"):
        st.info(
            "사내 데이터 연동 기능은 추후 제공 예정입니다. 현재는 CSV 업로드를 이용해주세요."
        )
    uploaded = st.file_uploader("CSV 파일 업로드", type=["csv"])
    if uploaded is None:
        st.info("계속하려면 CSV 파일을 업로드하세요.")
        return

    df, detected_encoding, read_error = _read_csv_with_fallback(uploaded)
    if df is None:
        st.error(f"CSV를 읽는 데 실패했습니다: {read_error}")
        return
    if detected_encoding and detected_encoding not in ("utf-8-sig", "utf-8"):
        st.info(f"CSV 인코딩을 `{detected_encoding}`로 감지해 로드했습니다.")

    st.session_state["uploaded_df"] = df
    st.caption(f"행: {len(df)} | 열: {len(df.columns)}")

    profile = None
    profile_error = None
    profile_context = None
    if objective_key == "model_training_readiness" and settings:
        profile_context = _build_profile_context(objective_key, settings, df)
        if settings.get("task_type") == "Auto Validate by History":
            profile, profile_error = _get_or_build_avh_profile(
                objective_key, settings, df
            )
        else:
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
                if settings.get("task_type") == "Auto Validate by History":
                    _render_avh_profile(profile)
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
            elif not effective_policy:
                st.info("정책을 로드할 수 없습니다.")
            else:
                report, report_error = _get_or_build_report(
                    profile_context, profile, effective_policy
                )
                if report_error:
                    st.error(f"평가에 실패했습니다: {report_error}")
                else:
                    decision = report.get("decision", "UNKNOWN")
                    if decision == "CONDITIONAL_GO":
                        st.markdown(
                            f"**결정:** <span style='color:#1f77b4;font-weight:700'>{decision}</span>",
                            unsafe_allow_html=True,
                        )
                    elif decision == "NO_GO":
                        st.markdown(
                            f"**결정:** <span style='color:#d62728;font-weight:700'>{decision}</span>",
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(f"**결정:** {decision}")
                    st.write(report.get("recommendation_text", ""))
                    checks = report.get("checks", [])
                    if checks:
                        st.dataframe(pd.DataFrame(checks), use_container_width=True)
                    if settings.get("task_type") == "Auto Validate by History":
                        st.divider()
                        st.markdown("**Auto Validate by History Result**")
                        _render_avh_profile(profile)
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

            if effective_policy and profile and profile_context:
                report, report_error = _get_or_build_report(
                    profile_context, profile, effective_policy
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
        task_type = st.selectbox(
            "모델 유형", ["classification", "regression", "QA Agent", "Auto Validate by History"]
        )
        target_column = None

        if columns:
            if task_type != "Auto Validate by History":
                target_column = st.selectbox("타깃 컬럼", columns)
            group_key = st.selectbox("그룹 키(선택)", ["(없음)"] + columns)
            time_column = st.selectbox("시간 컬럼(선택)", ["(없음)"] + columns)
        else:
            if task_type != "Auto Validate by History":
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

        settings_payload = {
            "task_type": task_type,
            "group_key": resolved_group_key,
            "time_column": resolved_time_column,
            "sampling_mode": sampling_mode,
            "sample_n": int(sample_n),
        }
        if task_type != "Auto Validate by History":
            settings_payload["target_column"] = target_column

        _save_settings(
            "model_training_readiness",
            settings_payload,
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


def render_policy_sidebar(policy: dict) -> None:
    policy_values = {
        "min_rows": policy.get("min_rows", "-"),
        "max_target_missing_rate": policy.get("max_target_missing_rate", "-"),
        "min_usable_features": policy.get("min_usable_features", "-"),
        "max_duplicate_row_rate": policy.get("max_duplicate_row_rate", "-"),
        "classification_minority_rate_warn": policy.get(
            "classification_minority_rate_warn", "-"
        ),
        "classification_minority_rate_fail": policy.get(
            "classification_minority_rate_fail", "-"
        ),
        "high_cardinality_warn_threshold": policy.get(
            "high_cardinality_warn_threshold", "-"
        ),
    }

    policy_markup = """
<style>
.policy-card {{
  border: 1px solid #e2e2e2;
  background: #fafafa;
  padding: 8px 10px;
  border-radius: 10px;
}}
.policy-row {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 6px 0;
  border-bottom: 1px dashed #dedede;
}}
.policy-row:last-child {{
  border-bottom: none;
}}
.policy-name {{
  font-weight: 600;
  font-size: 0.88rem;
  color: #2f2f2f;
}}
.policy-value {{
  font-size: 0.92rem;
  color: #111111;
  font-variant-numeric: tabular-nums;
  text-align: right;
}}
</style>
<div class="policy-card">
<div class="policy-row"><span class="policy-name">min_rows (최소 행 수)</span>
<span class="policy-value">{min_rows}</span></div>
<div class="policy-row"><span class="policy-name">max_target_missing_rate (타깃 결측 비율 상한)</span>
<span class="policy-value">{max_target_missing_rate}</span></div>
<div class="policy-row"><span class="policy-name">min_usable_features (사용 가능한 피처 최소 수)</span>
<span class="policy-value">{min_usable_features}</span></div>
<div class="policy-row"><span class="policy-name">max_duplicate_row_rate (중복 행 비율 상한)</span>
<span class="policy-value">{max_duplicate_row_rate}</span></div>
<div class="policy-row"><span class="policy-name">classification_minority_rate_warn (소수 클래스 비율 경고 기준)</span>
<span class="policy-value">{classification_minority_rate_warn}</span></div>
<div class="policy-row"><span class="policy-name">classification_minority_rate_fail (소수 클래스 비율 실패 기준)</span>
<span class="policy-value">{classification_minority_rate_fail}</span></div>
<div class="policy-row"><span class="policy-name">high_cardinality_warn_threshold (범주형 고카디널리티 경고 기준)</span>
<span class="policy-value">{high_cardinality_warn_threshold}</span></div>
</div>
"""
    st.sidebar.markdown(policy_markup.format_map(policy_values), unsafe_allow_html=True)



def _load_avh_criteria() -> dict:
    default_criteria = {
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
    if get_avh_evaluation_criteria is None:
        return default_criteria
    try:
        criteria = get_avh_evaluation_criteria()
        if isinstance(criteria, dict):
            merged = dict(default_criteria)
            merged.update(criteria)
            return merged
    except Exception:
        pass
    return default_criteria


def render_avh_policy_sidebar() -> None:
    criteria = _load_avh_criteria()
    policy_values = {
        "required_group_key": "required" if criteria.get("required_group_key") else "-",
        "required_time_column": "required" if criteria.get("required_time_column") else "-",
        "min_history_length_per_group": criteria.get("min_history_length_per_group", "-"),
        "datetime_parse_success_threshold": criteria.get(
            "datetime_parse_success_threshold", "-"
        ),
        "min_usable_history_groups": criteria.get("min_usable_history_groups", "-"),
        "min_history_ready_rate": criteria.get("min_history_ready_rate", "-"),
        "max_group_size_cv": criteria.get("max_group_size_cv", "-"),
        "max_duplicate_group_time_rate": criteria.get("max_duplicate_group_time_rate", "-"),
        "max_non_monotonic_group_rate": criteria.get("max_non_monotonic_group_rate", "-"),
        "max_null_share_in_latest_step": criteria.get("max_null_share_in_latest_step", "-"),
        "max_numeric_shift_score": criteria.get("max_numeric_shift_score", "-"),
        "max_target_flip_rate": criteria.get("max_target_flip_rate", "-"),
    }

    policy_markup = """
<style>
.policy-card {{
  border: 1px solid #e2e2e2;
  background: #fafafa;
  padding: 8px 10px;
  border-radius: 10px;
}}
.policy-row {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 6px 0;
  border-bottom: 1px dashed #dedede;
}}
.policy-row:last-child {{
  border-bottom: none;
}}
.policy-name {{
  font-weight: 600;
  font-size: 0.88rem;
  color: #2f2f2f;
}}
.policy-value {{
  font-size: 0.92rem;
  color: #111111;
  font-variant-numeric: tabular-nums;
  text-align: right;
}}
</style>
<div class="policy-card">
<div class="policy-row"><span class="policy-name">required_group_key (그룹 키 필수)</span>
<span class="policy-value">{required_group_key}</span></div>
<div class="policy-row"><span class="policy-name">required_time_column (시간 컬럼 필수)</span>
<span class="policy-value">{required_time_column}</span></div>
<div class="policy-row"><span class="policy-name">min_history_length_per_group (그룹별 최소 이력 길이)</span>
<span class="policy-value">{min_history_length_per_group}</span></div>
<div class="policy-row"><span class="policy-name">datetime_parse_success_threshold (날짜 파싱 성공률 기준)</span>
<span class="policy-value">{datetime_parse_success_threshold}</span></div>
<div class="policy-row"><span class="policy-name">min_usable_history_groups (사용 가능한 이력 그룹 최소 수)</span>
<span class="policy-value">{min_usable_history_groups}</span></div>
<div class="policy-row"><span class="policy-name">min_history_ready_rate (이력 준비율 최소 기준)</span>
<span class="policy-value">{min_history_ready_rate}</span></div>
<div class="policy-row"><span class="policy-name">max_group_size_cv (그룹 크기 변동계수 상한)</span>
<span class="policy-value">{max_group_size_cv}</span></div>
<div class="policy-row"><span class="policy-name">max_duplicate_group_time_rate (그룹-시간 중복 비율 상한)</span>
<span class="policy-value">{max_duplicate_group_time_rate}</span></div>
<div class="policy-row"><span class="policy-name">max_non_monotonic_group_rate (비단조 그룹 비율 상한)</span>
<span class="policy-value">{max_non_monotonic_group_rate}</span></div>
<div class="policy-row"><span class="policy-name">max_null_share_in_latest_step (최신 시점 결측 비율 상한)</span>
<span class="policy-value">{max_null_share_in_latest_step}</span></div>
<div class="policy-row"><span class="policy-name">max_numeric_shift_score (수치형 변화 점수 상한)</span>
<span class="policy-value">{max_numeric_shift_score}</span></div>
<div class="policy-row"><span class="policy-name">max_target_flip_rate (타깃 반전 비율 상한)</span>
<span class="policy-value">{max_target_flip_rate}</span></div>
</div>
"""
    st.sidebar.markdown("**AVH Policy Criteria**")
    st.sidebar.markdown(policy_markup.format_map(policy_values), unsafe_allow_html=True)
    selected_rules_df = st.session_state.get("avh_selected_rules_df")
    if isinstance(selected_rules_df, pd.DataFrame) and not selected_rules_df.empty:
        st.sidebar.markdown("**Selected AVH Rules**")
        cols = [c for c in ["column", "rule_type", "threshold"] if c in selected_rules_df.columns]
        st.sidebar.dataframe(
            _to_streamlit_safe_df(selected_rules_df[cols]),
            use_container_width=True,
            height=180,
        )


def run_o1_training(policy: dict | None) -> None:
    st.sidebar.header("설정")
    render_training_sidebar()
    current_settings = st.session_state["objective_settings"].get(
        "model_training_readiness", {}
    )
    task_type = current_settings.get("task_type")

    if task_type == "Auto Validate by History":
        render_avh_policy_sidebar()
    elif policy:
        st.sidebar.subheader("Validation Criteria")
        render_policy_sidebar(policy)
    else:
        st.sidebar.error("정책을 로드할 수 없습니다.")


def run_o2_evaluation() -> None:
    st.sidebar.header("설정")
    render_placeholder_sidebar("model_evaluation_readiness")


def run_o3_deployment() -> None:
    st.sidebar.header("설정")
    render_placeholder_sidebar("deployment_serving_readiness")


def run_o4_agent() -> list[str]:
    st.sidebar.markdown("PoC 리스트")
    selected = st.sidebar.multiselect(
        "PoC 선택",
        options=AGENT_POC_FEATURES,
        default=AGENT_POC_FEATURES[:1],
        key="o4_poc_selection",
    )
    if not selected:
        st.sidebar.info("PoC를 선택하면 해당 화면이 표시됩니다.")
    return selected


def render_selected_poc_pages(selected: list[str]) -> None:
    if not selected:
        st.info("왼쪽 PoC 리스트에서 항목을 선택하세요.")
        return

    for idx, poc in enumerate(selected):
        if len(selected) > 1:
            st.subheader(poc)

        page_key = POC_PAGES.get(poc)
        if page_key == "ontology_data_eval":
            render_agent_automation_page()
        else:
            st.warning("해당 PoC 화면은 준비되지 않았습니다.")

        if idx < len(selected) - 1:
            st.divider()


def render_agent_automation_page() -> None:
    if "o4_runs" not in st.session_state:
        st.session_state["o4_runs"] = []

    tabs = st.tabs(["Inputs", "Prompt", "Run & Compare", "History"])
    with tabs[0]:
        if st.button(
            "사내 데이터 불러오기 (Coming Soon)", key="internal_data_agent_inputs"
        ):
            st.info(
                "사내 데이터 연동 기능은 추후 제공 예정입니다. 현재는 CSV 업로드를 이용해주세요."
            )

        customer_id = st.text_input(
            "Customer ID (optional)", key="o4_customer_id"
        )
        st.caption(
            "Ontology mode: Nodes CSV + Edges CSV required. "
            "General DB mode: Table CSV required."
        )

        st.subheader("목표를 선택하세요")
        nodes_file = st.file_uploader(
            "Nodes CSV", type=["csv"], key="o4_nodes_csv"
        )
        edges_file = st.file_uploader(
            "Edges CSV", type=["csv"], key="o4_edges_csv"
        )
        context_file = st.file_uploader(
            "Context JSON (선택)", type=["json"], key="o4_context_json"
        )
        context_text = st.text_area(
            "Context JSON 붙여넣기 (선택)", key="o4_context_json_text"
        )

        if nodes_file is None:
            st.error("Nodes CSV 업로드가 필요합니다.")
        if edges_file is None:
            st.error("Edges CSV 업로드가 필요합니다.")

        if nodes_file is not None:
            nodes_df, nodes_encoding, nodes_error = _read_csv_with_fallback(nodes_file)
            if nodes_df is None:
                st.error(f"Nodes CSV를 읽는 데 실패했습니다: {nodes_error}")
                return
            st.session_state["o4_nodes_df"] = nodes_df
            st.caption(f"Nodes: {len(nodes_df)} rows x {len(nodes_df.columns)} cols")
            if nodes_encoding and nodes_encoding not in ("utf-8-sig", "utf-8"):
                st.info(f"Nodes CSV 인코딩 `{nodes_encoding}`으로 로드했습니다.")
            st.dataframe(nodes_df.head(10), use_container_width=True)

        if edges_file is not None:
            edges_df, edges_encoding, edges_error = _read_csv_with_fallback(edges_file)
            if edges_df is None:
                st.error(f"Edges CSV를 읽는 데 실패했습니다: {edges_error}")
                return
            st.session_state["o4_edges_df"] = edges_df
            st.caption(f"Edges: {len(edges_df)} rows x {len(edges_df.columns)} cols")
            if edges_encoding and edges_encoding not in ("utf-8-sig", "utf-8"):
                st.info(f"Edges CSV 인코딩 `{edges_encoding}`으로 로드했습니다.")
            st.dataframe(edges_df.head(10), use_container_width=True)

        context_raw = None
        if context_file is not None:
            context_raw, context_encoding, context_error = _decode_text_with_fallback(context_file)
            if context_raw is None:
                st.error(f"Context JSON을 읽는 데 실패했습니다: {context_error}")
                return
            if context_encoding and context_encoding not in ("utf-8-sig", "utf-8"):
                st.info(f"Context JSON 인코딩 `{context_encoding}`으로 로드했습니다.")
        elif context_text:
            context_raw = context_text
        if context_raw is not None:
            st.session_state["o4_context_json_raw"] = context_raw
            st.caption("Context JSON 로드됨")
            try:
                json.loads(context_raw)
                st.session_state["o4_context_json_valid"] = True
            except json.JSONDecodeError:
                st.session_state["o4_context_json_valid"] = False
                st.warning("Context JSON이 유효하지 않습니다. 자동 컨텍스트로 대체됩니다.")

        st.divider()
        st.subheader("목표를 선택하세요")
        table_file = st.file_uploader(
            "Table CSV", type=["csv"], key="o4_table_csv"
        )
        if table_file is None:
            st.error("Table CSV 업로드가 필요합니다.")
        else:
            table_df, table_encoding, table_error = _read_csv_with_fallback(table_file)
            if table_df is None:
                st.error(f"Table CSV를 읽는 데 실패했습니다: {table_error}")
                return
            st.session_state["o4_table_df"] = table_df
            st.caption(f"Table: {len(table_df)} rows x {len(table_df.columns)} cols")
            if table_encoding and table_encoding not in ("utf-8-sig", "utf-8"):
                st.info(f"Table CSV 인코딩 `{table_encoding}`으로 로드했습니다.")
            st.dataframe(table_df.head(10), use_container_width=True)

        if nodes_file is not None and edges_file is not None:
            filtered_nodes = _filter_by_customer_id(
                st.session_state["o4_nodes_df"], customer_id
            )
            filtered_edges = _filter_edges_by_nodes(
                st.session_state["o4_edges_df"], filtered_nodes
            )
            st.session_state["o4_nodes_filtered_df"] = filtered_nodes
            st.session_state["o4_edges_filtered_df"] = filtered_edges
            st.subheader("목표를 선택하세요")
            ontology_context = _build_ontology_context_markdown(
                filtered_nodes, filtered_edges
            )
            st.text_area(
                "Ontology Context",
                value=ontology_context,
                height=220,
                key="o4_ontology_context_preview",
            )

        if table_file is not None:
            filtered_table = _filter_by_customer_id(
                st.session_state["o4_table_df"], customer_id
            )
            st.session_state["o4_table_filtered_df"] = filtered_table
            st.subheader("목표를 선택하세요")
            table_context = _df_to_markdown(filtered_table.head(200))
            st.text_area(
                "Table Context",
                value=table_context,
                height=220,
                key="o4_table_context_preview",
            )

        st.session_state["o4_inputs_ready"] = bool(
            nodes_file is not None and edges_file is not None and table_file is not None
        )
    with tabs[1]:
        st.subheader("목표를 선택하세요")
        question = st.text_area("Question", key="o4_question", height=100)
        customer_id = st.session_state.get("o4_customer_id") or "(not specified)"

        with st.expander("Preview final prompts"):
            nodes_df = st.session_state.get("o4_nodes_filtered_df")
            edges_df = st.session_state.get("o4_edges_filtered_df")
            table_df = st.session_state.get("o4_table_filtered_df")

            ontology_context = ""
            if nodes_df is not None and edges_df is not None:
                ontology_context = _build_ontology_context_markdown(
                    nodes_df, edges_df
                )
            ontology_timeline = pd.DataFrame()
            if nodes_df is not None:
                ontology_timeline = _build_ontology_timeline_from_nodes(
                    nodes_df, st.session_state.get("o4_customer_id")
                )

            general_timeline = table_df if table_df is not None else pd.DataFrame()

            st.markdown("**Ontology Prompt**")
            st.text_area(
                "Ontology Prompt Preview",
                value=_build_ontology_prompt(
                    question or "(no question provided)",
                    customer_id,
                    ontology_timeline,
                    ontology_context or "(no ontology context)",
                ),
                height=220,
                key="o4_prompt_preview_ontology",
            )
            st.markdown("**General DB Prompt**")
            st.text_area(
                "General DB Prompt Preview",
                value=_build_general_db_prompt(
                    question or "(no question provided)",
                    customer_id,
                    general_timeline,
                ),
                height=220,
                key="o4_prompt_preview_table",
            )
    with tabs[2]:
        st.subheader("목표를 선택하세요")
        model = st.text_input("모델", value="llama3.1", key="o4_ollama_model")
        default_url = os.environ.get("OLLAMA_URL")
        if not default_url:
            default_url = "" if _is_cloud_runtime() else "http://localhost:11434"
        url = st.text_input(
            "Ollama URL", value=default_url, key="o4_ollama_url"
        )
        if _is_cloud_runtime() and not url:
            st.info(
                "Streamlit Cloud cannot reach localhost. Set a reachable Ollama URL (OLLAMA_URL)."
            )
        options_text = st.text_area(
            "Options JSON (선택)", key="o4_ollama_options", height=100
        )
        inputs_ready = bool(st.session_state.get("o4_inputs_ready"))
        run_clicked = st.button(
            "Run both (Ontology vs General DB)",
            key="o4_run_both",
            disabled=not inputs_ready,
        )

        if not inputs_ready:
            checklist = []
            if "o4_nodes_df" not in st.session_state:
                checklist.append("Nodes CSV")
            if "o4_edges_df" not in st.session_state:
                checklist.append("Edges CSV")
            if "o4_table_df" not in st.session_state:
                checklist.append("Table CSV")
            if checklist:
                st.warning(f"필수 업로드 필요: {', '.join(checklist)}")

        if run_clicked:
            if _is_cloud_runtime() and url.strip().startswith(
                (
                    "http://localhost",
                    "http://127.0.0.1",
                    "https://localhost",
                    "https://127.0.0.1",
                )
            ):
                st.error(
                    "Streamlit Cloud cannot reach localhost. Set a reachable Ollama URL."
                )
                return
            options = None
            if options_text.strip():
                try:
                    options = json.loads(options_text)
                except json.JSONDecodeError as exc:
                    st.error(f"Options JSON 파싱 실패: {exc}")
                    return

            question = st.session_state.get("o4_question", "")
            customer_id = st.session_state.get("o4_customer_id") or "(not specified)"

            nodes_df = st.session_state.get("o4_nodes_filtered_df")
            edges_df = st.session_state.get("o4_edges_filtered_df")
            table_df = st.session_state.get("o4_table_filtered_df")

            raw_context = st.session_state.get("o4_context_json_raw")
            ontology_context = ""
            if raw_context:
                try:
                    json.loads(raw_context)
                    ontology_context = raw_context
                except json.JSONDecodeError as exc:
                    st.warning(
                        f"Context JSON 파싱 실패: {exc}. Ontology CSV로 대체합니다."
                    )
            if not ontology_context and nodes_df is not None and edges_df is not None:
                ontology_context = _build_ontology_context_markdown(
                    nodes_df, edges_df
                )

            if not ontology_context or table_df is None:
                st.error("Ontology/General DB 컨텍스트를 먼저 준비하세요.")
                return

            ontology_timeline = _build_ontology_timeline_from_nodes(
                nodes_df, st.session_state.get("o4_customer_id")
            )
            general_timeline = table_df

            ontology_prompt = _build_ontology_prompt(
                question or "(no question provided)",
                customer_id,
                ontology_timeline,
                ontology_context,
            )
            general_prompt = _build_general_db_prompt(
                question or "(no question provided)",
                customer_id,
                general_timeline,
            )

            ontology_result = call_ollama_generate(
                ontology_prompt, model=model, url=url, options=options
            )
            general_result = call_ollama_generate(
                general_prompt, model=model, url=url, options=options
            )

            def _summary(result: dict) -> dict:
                response_text = result.get("response_text", "")
                return {
                    "ok": bool(result.get("ok")),
                    "response_length": len(response_text),
                    "error": result.get("error"),
                }

            st.session_state["o4_runs"].append(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "customer_id": customer_id,
                    "question": question,
                    "model": model,
                    "url": url,
                    "options": options,
                    "ontology_prompt": ontology_prompt,
                    "general_prompt": general_prompt,
                    "ontology_response_text": ontology_result.get("response_text", ""),
                    "general_response_text": general_result.get("response_text", ""),
                    "ontology_ok": ontology_result.get("ok"),
                    "general_ok": general_result.get("ok"),
                    "ontology_error": ontology_result.get("error"),
                    "general_error": general_result.get("error"),
                }
            )

            col_left, col_right = st.columns(2)
            with col_left:
                st.markdown("**Ontology Result**")
                if ontology_result.get("ok"):
                    st.text_area(
                        "Ontology Response",
                        value=ontology_result.get("response_text", ""),
                        height=220,
                    )
                else:
                    st.error(ontology_result.get("error", "Unknown error"))
                st.caption(_summary(ontology_result))

            with col_right:
                st.markdown("**General DB Result**")
                if general_result.get("ok"):
                    st.text_area(
                        "General DB Response",
                        value=general_result.get("response_text", ""),
                        height=220,
                    )
                else:
                    st.error(general_result.get("error", "Unknown error"))
                st.caption(_summary(general_result))
    with tabs[3]:
        runs = st.session_state.get("o4_runs", [])
        if not runs:
            st.info("아직 실행 이력이 없습니다.")
        else:
            options = list(reversed(runs))
            labels = []
            for idx, run in enumerate(options):
                question_text = run.get("question") or "(no question)"
                customer_label = run.get("customer_id") or "(no customer)"
                labels.append(
                    f"{run.get('timestamp')} | {customer_label} | {question_text[:60]} | #{len(options) - idx}"
                )

            selected = st.selectbox(
                "실행 이력 선택",
                options=list(range(len(options))),
                format_func=lambda i: labels[i],
            )
            run = options[selected]
            st.subheader("목표를 선택하세요")
            st.json(run)

            if st.button("Load this run settings", key="o4_load_run"):
                st.session_state["o4_question"] = run.get("question", "")
                st.session_state["o4_customer_id"] = run.get("customer_id", "")
                st.session_state["o4_ollama_model"] = run.get("model", "llama3.1")
                st.session_state["o4_ollama_url"] = run.get(
                    "url", "http://localhost:11434"
                )
                st.session_state["o4_ollama_options"] = json.dumps(
                    run.get("options") or {}, ensure_ascii=False, indent=2
                )
                st.success("질문과 Customer ID 입력을 복원했습니다.")


objective_key = st.session_state["objective"]

if objective_key:
    st.header(f"선택한 목표: {OBJECTIVE_LABELS.get(objective_key, objective_key)}")
    if st.button("목표 변경"):
        st.session_state["objective"] = None
        st.session_state.pop("uploaded_df", None)
        st.session_state["profile_cache"] = {}
        st.session_state["report_cache"] = {}
        st.rerun()

    if objective_key != "agent_automation_readiness":
        render_current_settings(objective_key)
    current_settings = st.session_state["objective_settings"].get(objective_key)

    policy_o1 = None
    if objective_key == "model_training_readiness":
        try:
            policy_o1 = load_policy(POLICY_PATH_O1)
        except Exception:
            policy_o1 = None

    if objective_key != "agent_automation_readiness":
        render_csv_section(objective_key, current_settings, policy_o1)

    if objective_key == "model_training_readiness":
        run_o1_training(policy_o1)
    elif objective_key == "model_evaluation_readiness":
        run_o2_evaluation()
    elif objective_key == "deployment_serving_readiness":
        run_o3_deployment()
    elif objective_key == "agent_automation_readiness":
        selected_pocs = run_o4_agent()
        render_selected_poc_pages(selected_pocs)
else:
    render_objective_selector()



