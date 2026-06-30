import sqlite3
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

DB_PATH = "db/aml_pipeline.db"
GITHUB_URL = "https://github.com/vasu-r2025/aml_rag_pipeline"

st.set_page_config(
    page_title="AML RAG Monitoring",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# STYLING
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
    :root {
        --accent: #5B3DF5;
        --accent-soft: #EEEAFE;
        --ink: #1A1A2E;
        --muted: #6B6B7B;
    }
    .block-container { padding-top: 2rem; }

    .metric-card {
        background: #FFFFFF;
        border: 1px solid #ECECF2;
        border-radius: 14px;
        padding: 18px 20px;
        box-shadow: 0 1px 2px rgba(20,20,40,0.03);
    }
    .metric-label {
        font-size: 0.78rem;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 1.9rem;
        font-weight: 700;
        color: var(--ink);
    }
    .metric-sub {
        font-size: 0.78rem;
        color: var(--accent);
        margin-top: 4px;
    }

    .alert-card {
        background: #FFFFFF;
        border: 1px solid #ECECF2;
        border-radius: 14px;
        padding: 16px 18px;
        margin-bottom: 12px;
    }
    .alert-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.03em;
    }
    .badge-critical { background: #FDE8E8; color: #C0392B; }
    .badge-high { background: #FFF1DA; color: #B8740A; }
    .badge-low { background: #E8F6EE; color: #1E8A4C; }

    .pill {
        display: inline-block;
        background: var(--accent-soft);
        color: var(--accent);
        border-radius: 999px;
        padding: 2px 10px;
        font-size: 0.72rem;
        margin-right: 6px;
        margin-top: 6px;
    }

    section[data-testid="stSidebar"] {
        background: #FAFAFC;
        border-right: 1px solid #ECECF2;
        min-width: 250px !important;
        max-width: 270px !important;
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data(ttl=30)
def load_transactions():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM transaction_results", conn)
    conn.close()
    return df


@st.cache_data(ttl=30)
def load_batch_runs():
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query(
            "SELECT * FROM batch_runs ORDER BY batch_number ASC", conn
        )
    except Exception:
        df = pd.DataFrame()
    conn.close()
    return df


def metric_card(label, value, sub=None):
    sub_html = f'<div class="metric-sub">{sub}</div>' if sub else ""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def risk_badge(risk_level):
    risk = str(risk_level).lower()
    if risk in ("critical",):
        return '<span class="alert-badge badge-critical">CRITICAL</span>'
    elif risk in ("high",):
        return '<span class="alert-badge badge-high">HIGH</span>'
    else:
        return '<span class="alert-badge badge-low">LOW</span>'


# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### AML RAG Pipeline")
    st.caption("Rule + RAG + LLM exception monitoring")

    page = st.radio(
        "Navigate",
        ["Overview", "Transaction Explorer", "Pipeline Performance"],
        label_visibility="collapsed",
    )

    st.divider()

    with st.expander("Cost Assumptions", expanded=False):
        cost_per_call = st.number_input(
            "Cost per LLM call (USD)",
            min_value=0.0001,
            max_value=0.05,
            value=0.0008,
            step=0.0001,
            format="%.4f",
        )
        st.caption("Approximates a Llama-3.3-70B call at typical Groq rates.")

    st.divider()

    st.markdown("**Simulate: force LLM for every transaction**")
    force_llm_sim = st.toggle("LLM-only mode (simulated)", value=False)
    st.caption(
        "When on, cost figures simulate what this batch would have cost "
        "if RAG were disabled and every transaction went to the LLM."
    )

    st.divider()

    if st.button("Reload data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.divider()

    with st.expander("About this project"):
        st.write(
            "A self-learning AML transaction monitoring pipeline. A rule "
            "engine and RAG retrieval resolve known patterns directly from "
            "ChromaDB. Transactions that don't match a known typology with "
            "enough confidence escalate to an LLM (Groq, Llama 3.3), which "
            "classifies the pattern and writes it back into the knowledge "
            "base so future similar transactions are resolved by RAG alone."
        )

    st.markdown(f"[View on GitHub]({GITHUB_URL})")


# ---------------------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------------------
df = load_transactions()
batch_df = load_batch_runs()

if df.empty:
    st.warning(
        "No data found in transaction_results. Run main.py first to populate the database."
    )
    st.stop()

total = len(df)
clean_count = (df["detection_status"].str.lower() == "clean").sum()
flagged_count = (df["detection_status"].str.lower() != "clean").sum()
llm_count = df["escalated_to_llm"].fillna(0).astype(int).sum()

st.title("AML Transaction Monitoring")
st.caption("Real-time view of rule, RAG, and LLM detection across the pipeline")

# ---------------------------------------------------------------------------
# PAGE: OVERVIEW
# ---------------------------------------------------------------------------
if page == "Overview":
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Total Transactions", f"{total:,}")
    with c2:
        metric_card("Clean", f"{clean_count:,}", sub="Resolved by rules")
    with c3:
        metric_card("Flagged High Risk", f"{flagged_count:,}")
    with c4:
        metric_card("Escalated to LLM", f"{llm_count:,}")

    st.write("")
    left, right = st.columns(2)

    with left:
        st.subheader("Risk Distribution")
        risk_order = ["Critical", "High", "Medium", "Low"]
        risk_counts = df["risk_level"].fillna("Unknown").value_counts()
        risk_colors = {
            "Critical": "#C0392B",
            "High": "#E08E0B",
            "Medium": "#3D7DF5",
            "Low": "#1E8A4C",
        }
        for level in risk_order:
            if level in risk_counts.index:
                count = risk_counts[level]
                pct = count / total * 100
                color = risk_colors.get(level, "#999")
                st.markdown(
                    f"""
                    <div style="margin-bottom:10px;">
                        <div style="display:flex; justify-content:space-between; font-size:0.85rem; margin-bottom:3px;">
                            <span>{level}</span><span style="color:{color}; font-weight:600;">{count:,} ({pct:.1f}%)</span>
                        </div>
                        <div style="background:#F0F0F5; border-radius:6px; height:8px;">
                            <div style="background:{color}; width:{pct}%; height:8px; border-radius:6px;"></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    with right:
        st.subheader("Typology Matches")
        typ_counts = (
            df[df["typology"].notna() & (df["typology"] != "")]["typology"]
            .value_counts()
            .reset_index()
        )
        typ_counts.columns = ["typology", "count"]
        if typ_counts.empty:
            st.info("No typology matches recorded yet.")
        else:
            fig_typ = px.bar(
                typ_counts,
                x="count",
                y="typology",
                orientation="h",
                color_discrete_sequence=["#5B3DF5"],
            )
            fig_typ.update_layout(
                yaxis=dict(categoryorder="total ascending"),
                plot_bgcolor="white",
                height=260,
                margin=dict(l=0, r=0, t=10, b=0),
                bargap=0.4,
            )
            st.plotly_chart(fig_typ, use_container_width=True)

# ---------------------------------------------------------------------------
# PAGE: TRANSACTION EXPLORER
# ---------------------------------------------------------------------------
elif page == "Transaction Explorer":
    st.subheader("Filter Transactions")

    f1, f2, f3, f4 = st.columns(4)
    with f1:
        risk_filter = st.multiselect(
            "Risk Level", options=sorted(df["risk_level"].dropna().unique().tolist())
        )
    with f2:
        status_filter = st.multiselect(
            "Detection Status",
            options=sorted(df["detection_status"].dropna().unique().tolist()),
        )
    with f3:
        typology_filter = st.multiselect(
            "Typology", options=sorted(df["typology"].dropna().unique().tolist())
        )
    with f4:
        llm_only = st.checkbox("Escalated to LLM only", value=False)

    filtered = df.copy()
    if risk_filter:
        filtered = filtered[filtered["risk_level"].isin(risk_filter)]
    if status_filter:
        filtered = filtered[filtered["detection_status"].isin(status_filter)]
    if typology_filter:
        filtered = filtered[filtered["typology"].isin(typology_filter)]
    if llm_only:
        filtered = filtered[filtered["escalated_to_llm"] == 1]

    st.caption(f"Showing {len(filtered):,} of {len(df):,} transactions")

    display_cols = [
        "id",
        "timestamp",
        "from_bank",
        "from_account",
        "to_bank",
        "to_account",
        "amount_paid",
        "payment_currency",
        "payment_format",
        "detection_status",
        "risk_level",
        "typology",
        "similarity",
        "rule_triggered",
        "escalated_to_llm",
    ]
    display_cols = [c for c in display_cols if c in filtered.columns]
    st.dataframe(filtered[display_cols], use_container_width=True, height=500)

    with st.expander("View LLM reasoning for a transaction"):
        llm_rows = filtered[filtered["escalated_to_llm"] == 1]
        if llm_rows.empty:
            st.info("No LLM-escalated transactions in the current filter.")
        else:
            selected_id = st.selectbox("Transaction ID", llm_rows["id"].tolist())
            reasoning = llm_rows.loc[
                llm_rows["id"] == selected_id, "llm_reasoning"
            ].values[0]
            st.write(reasoning if reasoning else "No reasoning recorded.")

# ---------------------------------------------------------------------------
# PAGE: PIPELINE PERFORMANCE
# ---------------------------------------------------------------------------
elif page == "Pipeline Performance":
    st.subheader("How Cost Drops as RAG Learns")

    if batch_df.empty:
        st.info("No batch run history yet. Run main.py to populate batch_runs.")
    else:
        latest = batch_df.iloc[-1]
        first = batch_df.iloc[0]

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            metric_card("Batches Run", f"{len(batch_df)}")
        with c2:
            metric_card(
                "Knowledge Base Size",
                f"{int(latest['knowledge_base_size'])}",
                sub=f"started at {int(first['knowledge_base_size'])}",
            )
        with c3:
            first_llm_pct = (
                first["llm_escalated"] / first["total_transactions"] * 100
                if first["total_transactions"]
                else 0
            )
            latest_llm_pct = (
                latest["llm_escalated"] / latest["total_transactions"] * 100
                if latest["total_transactions"]
                else 0
            )
            metric_card(
                "LLM Share, Latest Batch",
                f"{latest_llm_pct:.1f}%",
                sub=f"started at {first_llm_pct:.1f}%",
            )
        with c4:
            total_unknown = (
                batch_df["llm_unknown"].sum() if "llm_unknown" in batch_df else 0
            )
            metric_card("Genuinely Novel (Unclassified)", f"{int(total_unknown)}")

        st.write("")

        if force_llm_sim:
            st.warning(
                "LLM-only mode (simulated) is ON. The chart and savings below show what this "
                "run would have cost if every transaction had gone to the LLM, ignoring RAG."
            )

        # Per-batch LLM share trend
        batch_df["llm_pct"] = (
            batch_df["llm_escalated"] / batch_df["total_transactions"] * 100
        )
        batch_df["rag_pct"] = (
            batch_df["rag_resolved"] / batch_df["total_transactions"] * 100
        )

        fig_trend = go.Figure()
        fig_trend.add_trace(
            go.Bar(
                x=batch_df["batch_number"],
                y=batch_df["rag_pct"],
                name="Resolved by RAG/Rules",
                marker_color="#5B3DF5",
            )
        )
        fig_trend.add_trace(
            go.Bar(
                x=batch_df["batch_number"],
                y=batch_df["llm_pct"],
                name="Escalated to LLM",
                marker_color="#E08E0B",
            )
        )
        fig_trend.update_layout(
            barmode="stack",
            plot_bgcolor="white",
            xaxis_title="Batch",
            yaxis_title="% of batch",
            height=320,
            bargap=0.5,
            margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_trend, use_container_width=True)
        st.caption(
            "Each batch is 200 transactions. As the knowledge base grows from "
            "LLM-discovered patterns being written back, later batches need fewer LLM calls."
        )

        st.divider()

        # Cost comparison
        st.subheader("Cost Comparison")
        batch_df["actual_cost"] = batch_df["llm_escalated"] * cost_per_call
        batch_df["pure_llm_cost"] = batch_df["total_transactions"] * cost_per_call

        if force_llm_sim:
            display_cost_col = "pure_llm_cost"
            cost_label = "Simulated Cost (LLM-only)"
        else:
            display_cost_col = "actual_cost"
            cost_label = "Actual Cost (Rule + RAG + LLM)"

        fig_cost = go.Figure()
        fig_cost.add_trace(
            go.Scatter(
                x=batch_df["batch_number"],
                y=batch_df["pure_llm_cost"],
                name="If every transaction used the LLM",
                mode="lines+markers",
                line=dict(color="#C0392B", dash="dash"),
            )
        )
        fig_cost.add_trace(
            go.Scatter(
                x=batch_df["batch_number"],
                y=batch_df[display_cost_col],
                name=cost_label,
                mode="lines+markers",
                line=dict(color="#5B3DF5"),
            )
        )
        fig_cost.update_layout(
            plot_bgcolor="white",
            xaxis_title="Batch",
            yaxis_title="Estimated cost (USD)",
            height=320,
            margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_cost, use_container_width=True)

        total_pure_llm = batch_df["pure_llm_cost"].sum()
        total_actual = batch_df["actual_cost"].sum()
        savings = total_pure_llm - total_actual
        savings_pct = (savings / total_pure_llm * 100) if total_pure_llm > 0 else 0

        c1, c2, c3 = st.columns(3)
        with c1:
            metric_card("If LLM-Only (All Batches)", f"${total_pure_llm:,.4f}")
        with c2:
            metric_card("Actual Pipeline Cost", f"${total_actual:,.4f}")
        with c3:
            metric_card(
                "Total Savings", f"${savings:,.4f}", sub=f"{savings_pct:.1f}% lower"
            )

        with st.expander("View raw batch run log"):
            st.dataframe(batch_df, use_container_width=True)
