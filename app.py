"""
AI FMCG Sales Analyst — main Streamlit application.

Pipeline: upload -> profile -> quality check -> clean -> analytics engine
-> forecasting -> AI Analyst (Claude) -> management report.

Run with: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import tempfile
import os

from src.ingestion.loader import load_sales_file
from src.ingestion.profiler import profile_dataset
from src.cleaning.validation import run_quality_checks
from src.cleaning.cleaner import clean_dataset
from src.analytics.engine import run_analytics
from src.forecasting.models import train_and_forecast
from src.reporting.charts import revenue_trend_chart, revenue_by_dimension, pareto_chart, forecast_chart
from src.reporting.reports import generate_management_report
from src.ai.orchestrator import SalesAnalystAgent

st.set_page_config(page_title="AI FMCG Sales Analyst", layout="wide")
st.title("🥩 AI FMCG Sales Analyst")

uploaded = st.file_uploader("Upload sales data (CSV or Excel)", type=["csv", "xlsx", "xls"])

if not uploaded:
    st.info("Upload a sales export to begin. Any FMCG POS/ERP export works — column names are auto-detected.")
    st.stop()

# Persist the uploaded file to disk so our loader (which takes a path) can read it
tmp_path = os.path.join(tempfile.gettempdir(), uploaded.name)
with open(tmp_path, "wb") as f:
    f.write(uploaded.getbuffer())

raw_df = load_sales_file(tmp_path)

(
    tab_overview, tab_quality, tab_analytics, tab_forecast, tab_ai, tab_report
) = st.tabs([
    "📋 Data Overview", "🧹 Data Quality", "📈 Sales Analytics",
    "🔮 Forecasting", "🤖 AI Analyst", "📄 Management Report",
])

# ---------------- Data Overview ----------------
with tab_overview:
    profile = profile_dataset(raw_df)
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{profile['row_count']:,}")
    c2.metric("Columns", profile["column_count"])
    c3.metric("Duplicate rows", profile["duplicate_rows"])

    st.subheader("Detected schema")
    st.json(profile["schema"]["available"])
    if profile["schema"]["missing_roles"]:
        st.caption(f"Not detected (analyses using these will be skipped): {', '.join(profile['schema']['missing_roles'])}")

    st.subheader("Preview")
    st.dataframe(raw_df.head(20), use_container_width=True)

# ---------------- Data Quality ----------------
with tab_quality:
    quality_report = run_quality_checks(raw_df)
    st.metric("Data Quality Score", f"{quality_report['quality_score']}/100")

    if quality_report["issues"]:
        st.dataframe(pd.DataFrame(quality_report["issues"]), use_container_width=True)
    else:
        st.success("No quality issues detected.")

    st.subheader("Cleaning actions applied")
    clean_result = clean_dataset(raw_df)
    df = clean_result["cleaned_df"]

    if clean_result["actions"]:
        st.dataframe(pd.DataFrame(clean_result["actions"]), use_container_width=True)
    else:
        st.success("No automated cleaning was necessary.")

    if not clean_result["rejected_rows"].empty:
        with st.expander(f"{len(clean_result['rejected_rows'])} row(s) removed — inspect"):
            st.dataframe(clean_result["rejected_rows"], use_container_width=True)

# Cache analytics across tabs
if "df_clean" not in st.session_state or not st.session_state["df_clean"].equals(df):
    st.session_state["df_clean"] = df
    st.session_state["analytics"] = run_analytics(df)

analytics = st.session_state["analytics"]

# ---------------- Sales Analytics ----------------
with tab_analytics:
    st.subheader("Core KPIs")
    core = analytics["core_kpis"]
    cols = st.columns(min(len(core), 4) or 1)
    for i, (name, value) in enumerate(core.items()):
        display = f"R {value:,.2f}" if any(k in name for k in ("revenue", "value", "price")) else f"{value:,.2f}"
        cols[i % 4].metric(name.replace("_", " ").title(), display)

    if analytics["profitability"]:
        st.subheader("Profitability")
        p = analytics["profitability"]
        pcols = st.columns(4)
        pcols[0].metric("Gross Profit", f"R {p['gross_profit']:,.2f}")
        pcols[1].metric("Gross Margin", f"{p.get('gross_margin_pct', 0):.1f}%")

    if analytics["target_performance"]["available"]:
        st.subheader("Target Performance")
        t = analytics["target_performance"]
        tcols = st.columns(4)
        tcols[0].metric("Actual", f"R {t['actual']:,.2f}")
        tcols[1].metric("Target", f"R {t['target']:,.2f}")
        tcols[2].metric("Variance %", f"{t['variance_pct']:.1f}%")
        tcols[3].metric("Achievement", f"{t['achievement_pct']:.1f}%")

    if analytics["monthly_sales"] is not None:
        st.subheader("Revenue Trend")
        fig = revenue_trend_chart(analytics["monthly_sales"])
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    if analytics["products"] is not None:
        st.subheader("Top Products")
        fig = revenue_by_dimension(analytics["products"], analytics["products"].columns[0])
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    if analytics["pareto_products"] is not None:
        st.subheader("Pareto Analysis — Products")
        fig = pareto_chart(analytics["pareto_products"], analytics["pareto_products"].columns[0])
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    for label, key in [("Customers", "customers"), ("Stores", "stores"), ("Regions", "regions")]:
        if analytics[key] is not None:
            st.subheader(label)
            st.dataframe(analytics[key].head(20), use_container_width=True)

    if analytics["anomalies"]["available"]:
        st.subheader("Anomalies")
        st.write(f"{analytics['anomalies']['anomaly_count']} unusual daily-revenue observation(s) flagged.")
        if analytics["anomalies"]["anomalies"]:
            st.dataframe(pd.DataFrame(analytics["anomalies"]["anomalies"]), use_container_width=True)

# ---------------- Forecasting ----------------
with tab_forecast:
    st.subheader("Revenue Forecast")
    horizon = st.slider("Forecast horizon (days)", 7, 60, 30)
    forecast_result = train_and_forecast(df, horizon=horizon)

    if forecast_result["available"]:
        e = forecast_result["evaluation"]
        c1, c2 = st.columns(2)
        c1.metric("Model MAE", f"R {e['model']['mae']:,.0f}")
        c2.metric("Seasonal-naive baseline MAE", f"R {e['seasonal_naive_baseline']['mae']:,.0f}")
        st.caption("Model is only worth trusting if its MAE beats the baseline above.")

        fig = forecast_chart(analytics["daily_sales"], forecast_result["forecast"])
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        st.metric(f"Forecast total (next {horizon} days)", f"R {forecast_result['forecast_total']:,.2f}")
    else:
        st.warning(forecast_result["reason"])

# ---------------- AI Analyst ----------------
with tab_ai:
    st.subheader("Ask the AI Sales Analyst")
    st.caption("Claude interprets validated evidence from the analytics engine above — it never sees raw rows or invents numbers.")

    api_key_present = bool(st.session_state.get("anthropic_key_set"))
    question = st.text_input("Ask a question, e.g. 'Why did revenue change?' or 'Which products lead sales?'")

    if question:
        agent = SalesAnalystAgent(df)
        try:
            response = agent.answer_question(question)
            st.markdown(f"**Intent detected:** `{response['intent']}`")
            st.write(response["answer"])
        except (ImportError, ValueError) as e:
            st.error(f"AI Analyst unavailable: {e}. Set ANTHROPIC_API_KEY in your .env file.")

# ---------------- Management Report ----------------
with tab_report:
    st.subheader("Automated Management Report")
    report_text = generate_management_report(analytics)
    st.markdown(report_text)
    st.download_button("Download report (.md)", report_text, file_name="management_report.md")
