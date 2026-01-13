import json
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from core.evaluator import evaluate_o1
from core.llm_client import call_ollama_generate
from core.policy import load_policy
from core.profiler import build_profile_o1

st.set_page_config(page_title="AI 품질 검증 Agent", layout="wide")

st.title("AI 품질 검증 Agent")

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
    {"key": "model_training_readiness", "label": "모델 학습 데이터 품질"},
    {"key": "model_evaluation_readiness", "label": "모델 평가 데이터 품질"},
    {"key": "deployment_serving_readiness", "label": "배포/서빙 Readiness"},
    {"key": "agent_automation_readiness", "label": "Agent/Automation Readiness"},
]

OBJECTIVE_LABELS = {item["key"]: item["label"] for item in OBJECTIVES}

AGENT_POC_FEATURES = [
    "온톨로지 데이터 평가 PoC",
]

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


def _df_to_markdown(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except Exception:
        return df.to_csv(index=False)


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
    if st.button("사내 데이터 불러오기 (Coming Soon)", key="internal_data_main"):
        st.info(
            "사내 데이터 연동 기능은 추후 제공 예정입니다. 현재는 CSV 업로드를 이용해주세요."
        )
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
        task_type = st.selectbox("모델 유형", ["classification", "regression"])

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


def run_o1_training(policy: dict | None) -> None:
    st.sidebar.header("설정")
    render_training_sidebar()
    if policy:
        st.sidebar.subheader("검증 기준값")
        render_policy_sidebar(policy)
    else:
        st.sidebar.error("정책을 로드할 수 없습니다.")


def run_o2_evaluation() -> None:
    st.sidebar.header("설정")
    render_placeholder_sidebar("model_evaluation_readiness")


def run_o3_deployment() -> None:
    st.sidebar.header("설정")
    render_placeholder_sidebar("deployment_serving_readiness")


def run_o4_agent() -> None:
    st.sidebar.button("기능", key="o4_feature_button")
    st.sidebar.markdown("PoC 리스트")
    for item in AGENT_POC_FEATURES:
        st.sidebar.markdown(f"- {item}")


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

        st.subheader("Ontology 입력")
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
            nodes_df = pd.read_csv(nodes_file)
            st.session_state["o4_nodes_df"] = nodes_df
            st.caption(f"Nodes: {len(nodes_df)} rows × {len(nodes_df.columns)} cols")
            st.dataframe(nodes_df.head(10), use_container_width=True)

        if edges_file is not None:
            edges_df = pd.read_csv(edges_file)
            st.session_state["o4_edges_df"] = edges_df
            st.caption(f"Edges: {len(edges_df)} rows × {len(edges_df.columns)} cols")
            st.dataframe(edges_df.head(10), use_container_width=True)

        context_raw = None
        if context_file is not None:
            context_raw = context_file.read().decode("utf-8")
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
        st.subheader("General DB 입력")
        table_file = st.file_uploader(
            "Table CSV", type=["csv"], key="o4_table_csv"
        )
        if table_file is None:
            st.error("Table CSV 업로드가 필요합니다.")
        else:
            table_df = pd.read_csv(table_file)
            st.session_state["o4_table_df"] = table_df
            st.caption(f"Table: {len(table_df)} rows × {len(table_df.columns)} cols")
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
            st.subheader("Ontology Context Preview")
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
            st.subheader("General Table Context Preview")
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
        st.subheader("Prompt 입력")
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
        st.subheader("Run & Compare")
        model = st.text_input("모델", value="llama3.1", key="o4_ollama_model")
        url = st.text_input(
            "Ollama URL", value="http://localhost:11434", key="o4_ollama_url"
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
            st.subheader("Run Details")
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
        render_agent_automation_page()
        run_o4_agent()
else:
    render_objective_selector()
