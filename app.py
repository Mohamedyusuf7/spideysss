import streamlit as st
import pandas as pd
import numpy as np
import os
import io
import matplotlib.pyplot as plt
import seaborn as sns
from agent import run_autonomous_ml_pipeline, generate_dataset_intelligence_report

# --- PAGE SETUP ---
st.set_page_config(
    page_title="AutoData Scientist | Autonomous ML Hub",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- BLACK & GOLD THEME CSS ---
st.markdown("""
<style>
    .stApp {
        background-color: #0b0c10;
        color: #e0e0e0;
        font-family: 'Segoe UI', Roboto, sans-serif;
    }
    
    [data-testid="stSidebar"] {
        background-color: #0f1117;
        border-right: 1px solid #232733;
    }
    
    h1, h2, h3, h4 {
        color: #d4af37 !important;
        font-weight: 700;
        letter-spacing: 0.5px;
    }

    .report-card {
        background-color: #12141a;
        border: 1px solid #232733;
        border-radius: 10px;
        padding: 24px;
        margin: 16px 0;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
        line-height: 1.8;
        font-size: 15px;
    }
    .report-card h1 {
        color: #ffffff !important;
        font-size: 28px;
        border-bottom: 2px solid #d4af37;
        padding-bottom: 8px;
        margin-bottom: 18px;
    }
    .report-card h2 {
        color: #d4af37 !important;
        font-size: 20px;
        margin-top: 20px;
        margin-bottom: 8px;
    }
    .report-card p, .report-card li {
        color: #d1d5db;
    }

    .glow-badge {
        display: inline-block;
        padding: 4px 16px;
        background: rgba(212, 175, 55, 0.08);
        border: 1px solid #d4af37;
        border-radius: 20px;
        color: #d4af37;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1.2px;
        margin-bottom: 12px;
    }
    
    .metric-card {
        background-color: #151821;
        border: 1px solid #2a2e3d;
        border-radius: 10px;
        padding: 18px 12px;
        text-align: center;
        margin-bottom: 12px;
    }
    .metric-card:hover {
        border-color: #d4af37;
    }
    .metric-val {
        font-size: 26px;
        font-weight: 800;
        color: #d4af37;
        margin-bottom: 4px;
    }
    .metric-lbl {
        font-size: 11px;
        color: #8a8f9d;
        font-weight: 700;
        letter-spacing: 0.8px;
    }

    .info-banner {
        background-color: #131722;
        border-left: 4px solid #d4af37;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 14px 0;
        font-size: 14px;
        color: #d1d5db;
    }

    .exec-log-item {
        background-color: #151821;
        border-left: 3px solid #d4af37;
        padding: 10px 14px;
        margin-bottom: 8px;
        border-radius: 0 6px 6px 0;
        font-size: 13px;
        color: #e0e0e0;
    }

    .success-card {
        background: linear-gradient(135deg, rgba(212, 175, 55, 0.15) 0%, rgba(15, 17, 23, 0.8) 100%);
        border: 1px solid #d4af37;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin-bottom: 20px;
    }

    div.stButton > button:first-child {
        background: linear-gradient(135deg, #d4af37 0%, #aa820a 100%);
        color: #0b0c10;
        font-weight: 700;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-size: 15px;
        box-shadow: 0 4px 15px rgba(212, 175, 55, 0.25);
    }
    div.stButton > button:first-child:hover {
        background: linear-gradient(135deg, #f3e5ab 0%, #d4af37 100%);
        color: #000;
    }

    div.stDownloadButton > button {
        background: #151821 !important;
        color: #d4af37 !important;
        border: 1px solid #d4af37 !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        padding: 10px 20px !important;
        width: 100%;
    }
    div.stDownloadButton > button:hover {
        background: #d4af37 !important;
        color: #0b0c10 !important;
    }
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE INITIALIZATION ---
if "raw_df" not in st.session_state:
    st.session_state.raw_df = None
if "clean_df" not in st.session_state:
    st.session_state.clean_df = None
if "agent_result" not in st.session_state:
    st.session_state.agent_result = None
if "file_name" not in st.session_state:
    st.session_state.file_name = "dataset"
if "editorial_report" not in st.session_state:
    st.session_state.editorial_report = None
if "transformation_logs" not in st.session_state:
    st.session_state.transformation_logs = []
if "nav_view" not in st.session_state:
    st.session_state.nav_view = "Initial Report"

# --- SIDEBAR: DATA INGESTION & MODULE NAVIGATION ---
with st.sidebar:
    st.markdown("###  Dataset Ingestion")
    uploaded_file = st.file_uploader("Drag & drop a CSV or Excel file", type=["csv", "xlsx"])

    if uploaded_file is not None:
        os.makedirs("data", exist_ok=True)
        temp_path = os.path.join("data", f"temp_{uploaded_file.name}")
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        try:
            if uploaded_file.name.endswith(".csv"):
                try:
                    df = pd.read_csv(temp_path, encoding="utf-8")
                except UnicodeDecodeError:
                    try:
                        df = pd.read_csv(temp_path, encoding="latin1")
                    except UnicodeDecodeError:
                        df = pd.read_csv(temp_path, encoding="ISO-8859-1")
            else:
                df = pd.read_excel(temp_path)
            
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
            
            if st.session_state.raw_df is None or st.session_state.file_name != uploaded_file.name:
                st.session_state.raw_df = df
                st.session_state.clean_df = df.copy()
                st.session_state.file_name = uploaded_file.name
                st.session_state.editorial_report = None
                st.session_state.nav_view = "Initial Report"
                st.session_state.transformation_logs = ["Initial raw dataset loaded."]

            st.success(f" `{uploaded_file.name}` ingested")
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")

    st.markdown("---")
    st.markdown("###  Launchpad Navigation")
    
    nav_options = [
        "📰 Initial Dataset Intelligence Report",
        "1.  Module 01 — Validation & Health",
        "2.  Module 02 — Intelligence & Profiling",
        "3.  Module 03 — Data Transformation",
        "4.  Module 04 — AutoML Model Engine"
    ]
    
    default_idx = 0 if st.session_state.nav_view == "Initial Report" else 1
    selected_module = st.radio("Navigation", nav_options, index=default_idx, label_visibility="collapsed")

# --- MAIN SCREEN LOGIC ---
if st.session_state.raw_df is None:
    st.markdown('<div class="glow-badge"> AUTODATA AGENT V2.0 • LAUNCHPAD</div>', unsafe_allow_html=True)
    st.markdown("#  AutoData SCIENTIST")
    st.markdown("<p style='color: #8a8f9d; font-size: 16px; margin-top: -10px;'>Venom-Strike Intelligence Hub — Profile, Clean & Train AutoML Models</p>", unsafe_allow_html=True)
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card" style="text-align: left; padding: 20px;">
            <h4 style="margin: 0 0 10px 0;"> Automated Profiling</h4>
            <p style="color: #a0a5b5; font-size: 13px; margin: 0;">Instantly profile datasets with full statistical summaries, missing value detection, and outlier analysis.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card" style="text-align: left; padding: 20px;">
            <h4 style="margin: 0 0 10px 0;">⚡ AI Decision Engine</h4>
            <p style="color: #a0a5b5; font-size: 13px; margin: 0;">Leverage Gemini Agent for deep dataset reflection, autonomous repair, and automated cleaning code.</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card" style="text-align: left; padding: 20px;">
            <h4 style="margin: 0 0 10px 0;">🚀 Baseline AutoML</h4>
            <p style="color: #a0a5b5; font-size: 13px; margin: 0;">Automatically evaluate classification or regression pipelines with instant model artifacts.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-banner" style="margin-top: 30px;">
        👉 <b>Upload a CSV or Excel file in the sidebar to launch Mission Control.</b>
    </div>
    """, unsafe_allow_html=True)

else:
    df = st.session_state.raw_df
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = int(df.isnull().sum().sum())
    missing_pct = (missing_cells / total_cells * 100) if total_cells > 0 else 0
    duplicates = int(df.duplicated().sum())
    memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
    health_score = max(0, int(100 - (missing_pct * 2) - (duplicates * 5)))

    # =========================================================================
    # VIEW: DYNAMIC AI-GENERATED REPORT (SHOWN FIRST ON UPLOAD)
    # =========================================================================
    if "Initial Dataset" in selected_module or st.session_state.nav_view == "Initial Report":
        if st.session_state.editorial_report is None:
            summary_info = f"""
Filename: {st.session_state.file_name}
Total Rows: {df.shape[0]}
Total Columns: {df.shape[1]}
Missing Cells: {missing_cells} ({missing_pct:.2f}%)
Duplicate Rows: {duplicates}
Numeric Columns: {', '.join(num_cols) if num_cols else 'None'}
Categorical Columns: {', '.join(cat_cols) if cat_cols else 'None'}
Data Quality Score: {health_score}/100
"""
            with st.spinner("🤖 Lead AI Scientist analyzing dataset semantics and generating executive audit report..."):
                report_md = generate_dataset_intelligence_report(summary_info, df.head(5).to_string())
                st.session_state.editorial_report = report_md

        st.markdown(f'<div class="report-card">{st.session_state.editorial_report}</div>', unsafe_allow_html=True)

        col_d1, col_d2 = st.columns([1, 1])
        with col_d1:
            st.download_button(
                label="📄 Download Full Intelligence Audit Report (.md / .txt)",
                data=st.session_state.editorial_report,
                file_name=f"{st.session_state.file_name}_audit_report.md",
                mime="text/markdown"
            )
        with col_d2:
            if st.button("🚀 Proceed to Launchpad Modules"):
                st.session_state.nav_view = "Modules"
                st.rerun()

    # =========================================================================
    # MODULE 01: DATA VALIDATION & HEALTH
    # =========================================================================
    elif "Module 01" in selected_module:
        st.markdown('<div class="glow-badge">⚡ MISSION CONTROL • MODULE 01</div>', unsafe_allow_html=True)
        st.title("📋 Module 01 — Data Validation & Health")

        st.markdown(f'<div class="glow-badge">✔ Data Quality Score: {health_score}/100</div>', unsafe_allow_html=True)

        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{df.shape[0]}</div><div class="metric-lbl">TOTAL ROWS</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{df.shape[1]}</div><div class="metric-lbl">TOTAL COLUMNS</div></div>', unsafe_allow_html=True)
        with m3:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{missing_pct:.1f}%</div><div class="metric-lbl">MISSING DATA</div></div>', unsafe_allow_html=True)
        with m4:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{duplicates}</div><div class="metric-lbl">DUPLICATES</div></div>', unsafe_allow_html=True)
        with m5:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{memory_mb:.1f} MB</div><div class="metric-lbl">MEMORY SIZE</div></div>', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="info-banner">
            🔍 <b>[Validation Report]</b> {missing_cells} missing cells detected ({missing_pct:.1f}% of entire dataset).
        </div>
        """, unsafe_allow_html=True)

        with st.expander("📄 Raw Data Preview (First 10 Rows)", expanded=True):
            st.dataframe(df.head(10), use_container_width=True)

    # =========================================================================
    # MODULE 02: INTELLIGENCE & PROFILING (WITH SEPARATE AI INSIGHTS)
    # =========================================================================
    elif "Module 02" in selected_module:
        st.markdown('<div class="glow-badge">⚡ MISSION CONTROL • MODULE 02</div>', unsafe_allow_html=True)
        st.title("📊 Module 02 — Exploratory Profiling & AI Analysis")

        tab_meta, tab_dist, tab_corr, tab_ai = st.tabs([
            "📋 Column Meta",
            "📈 Distributions",
            "🔗 Correlations",
            "⚡ AI Analysis"
        ])

        with tab_meta:
            meta_rows = []
            for col in df.columns:
                series = df[col]
                missing_cnt = series.isnull().sum()
                missing_p = (missing_cnt / len(df)) * 100
                unique_cnt = series.nunique()
                sample_vals = ", ".join([str(x) for x in series.dropna().unique()[:3]])

                meta_rows.append({
                    "Column": col,
                    "Type": str(series.dtype),
                    "Missing": missing_cnt,
                    "Missing %": f"{missing_p:.1f}%",
                    "Unique": unique_cnt,
                    "Sample Values": sample_vals
                })

            st.dataframe(pd.DataFrame(meta_rows), use_container_width=True)

        with tab_dist:
            if num_cols:
                selected_num = st.selectbox("Select Numeric Feature to Plot:", options=num_cols)
                fig, ax = plt.subplots(figsize=(8, 3.5))
                fig.patch.set_facecolor('#151821')
                ax.set_facecolor('#151821')
                sns.histplot(df[selected_num].dropna(), kde=True, color='#d4af37', ax=ax)
                ax.tick_params(colors='#e0e0e0')
                ax.xaxis.label.set_color('#d4af37')
                ax.yaxis.label.set_color('#d4af37')
                st.pyplot(fig)
            elif cat_cols:
                selected_cat = st.selectbox("Select Feature:", options=cat_cols)
                fig, ax = plt.subplots(figsize=(8, 3.5))
                fig.patch.set_facecolor('#151821')
                ax.set_facecolor('#151821')
                df[selected_cat].value_counts().head(10).plot(kind='bar', color='#d4af37', ax=ax)
                ax.tick_params(colors='#e0e0e0')
                st.pyplot(fig)

        with tab_corr:
            num_df = df.select_dtypes(include=[np.number])
            if num_df.shape[1] > 1:
                fig, ax = plt.subplots(figsize=(8, 5))
                fig.patch.set_facecolor('#151821')
                ax.set_facecolor('#151821')
                sns.heatmap(num_df.corr(), annot=True, cmap='magma', ax=ax)
                ax.tick_params(colors='#e0e0e0')
                st.pyplot(fig)
            else:
                st.info("Need at least 2 numeric columns for correlation heatmap.")

        with tab_ai:
            if st.button("⚡ Generate AI Insight Report"):
                summary_info = f"Dataset: {st.session_state.file_name}\nRows: {df.shape[0]}, Cols: {df.shape[1]}\nColumns: {', '.join(df.columns)}"
                with st.spinner("🤖 Generating LLM synthesis report..."):
                    st.session_state.ai_insights = generate_dataset_intelligence_report(summary_info, df.head(5).to_string())
            
            if "ai_insights" in st.session_state:
                st.markdown(f'<div class="report-card">{st.session_state.ai_insights}</div>', unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="info-banner">
                    👉 Click the button above to generate an LLM dataset synthesis report.
                </div>
                """, unsafe_allow_html=True)

    # =========================================================================
    # MODULE 03: DATA TRANSFORMATION & SHAPE PIPELINE
    # =========================================================================
    elif "Module 03" in selected_module:
        st.markdown('<div class="glow-badge">⚡ MISSION CONTROL • MODULE 03</div>', unsafe_allow_html=True)
        st.title("🧹 Module 03 — Data Filtering & Preprocessing")

        c1, c2 = st.columns([1, 1])
        with c1:
            st.subheader("Preprocessing Pipeline Options")
            drop_dups = st.checkbox("🗑️ Drop duplicate rows", value=True)
            drop_null_rows = st.checkbox("🗑️ Drop fully-null rows", value=False)
            drop_const = st.checkbox("🗑️ Drop constant (zero variance) columns", value=False)
            drop_null_cols = st.checkbox("🗑️ Drop fully-null columns", value=False)
            drop_id_cols = st.checkbox("🗑️ Drop high-cardinality ID columns", value=False)
            impute_vals = st.checkbox("🛠️ Impute missing values (median/mode)", value=True)
            remove_outliers = st.checkbox("📊 Remove numeric outliers (IQR)", value=False)

            if st.button("🚀 Apply Transformations"):
                clean = df.copy()
                logs = []
                if drop_dups and duplicates > 0:
                    clean = clean.drop_duplicates()
                    logs.append(f"Dropped {duplicates} duplicate rows.")
                if drop_null_rows:
                    clean = clean.dropna(how='all')
                    logs.append("Dropped fully-null rows.")
                if drop_null_cols:
                    clean = clean.dropna(axis=1, how='all')
                    logs.append("Dropped fully-null columns.")
                if drop_const:
                    clean = clean.loc[:, clean.nunique() > 1]
                    logs.append("Dropped zero-variance columns.")
                if drop_id_cols:
                    id_cols = [c for c in clean.columns if "id" in c.lower() or "name" in c.lower() or "ticket" in c.lower()]
                    clean = clean.drop(columns=id_cols, errors='ignore')
                    logs.append(f"Dropped ID columns: {', '.join(id_cols)}")
                if impute_vals:
                    imp_count = int(clean.isnull().sum().sum())
                    for col in clean.columns:
                        if pd.api.types.is_numeric_dtype(clean[col]):
                            clean[col] = clean[col].fillna(clean[col].median())
                        else:
                            clean[col] = clean[col].fillna(clean[col].mode()[0] if not clean[col].mode().empty else "Unknown")
                    logs.append(f"Imputed {imp_count} missing values using median/mode.")
                
                if not logs:
                    logs.append("Dataset already clean; no modifications required.")
                
                st.session_state.clean_df = clean
                st.session_state.transformation_logs = logs
                st.success("✅ Transformations applied successfully!")

        with c2:
            st.subheader("Shape Transformation")
            s1, s2 = st.columns(2)
            with s1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-val">{df.shape[0]} × {df.shape[1]}</div>
                    <div class="metric-lbl">ORIGINAL DATASET</div>
                </div>
                """, unsafe_allow_html=True)
            with s2:
                clean_shape = st.session_state.clean_df.shape
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-val">{clean_shape[0]} × {clean_shape[1]}</div>
                    <div class="metric-lbl">CLEANED DATASET</div>
                </div>
                """, unsafe_allow_html=True)

            st.subheader("📑 Execution Log")
            for log in st.session_state.transformation_logs:
                st.markdown(f'<div class="exec-log-item">✔ {log}</div>', unsafe_allow_html=True)

            csv_bytes = st.session_state.clean_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Cleaned CSV",
                data=csv_bytes,
                file_name=f"cleaned_{st.session_state.file_name}.csv",
                mime="text/csv"
            )

        st.markdown("---")
        with st.expander("📄 Processed Data Preview", expanded=False):
            st.dataframe(st.session_state.clean_df.head(10), use_container_width=True)

    # =========================================================================
    # MODULE 04: AUTOML MODEL ENGINE & SEPARATE CHARTS
    # =========================================================================
    elif "Module 04" in selected_module:
        st.markdown('<div class="glow-badge">⚡ MISSION CONTROL • MODULE 04</div>', unsafe_allow_html=True)
        st.title("🚀 Module 04 — AutoML Model Engine")

        active_df = st.session_state.clean_df
        all_cols = active_df.columns.tolist()

        target_col = st.selectbox(
            "🎯 Select Target Variable:",
            options=all_cols,
            index=0
        )

        target_series = active_df[target_col]
        target_type = str(target_series.dtype)
        unique_vals = target_series.nunique()
        is_classification = unique_vals <= 20 or not pd.api.types.is_numeric_dtype(target_series)
        task_type = "Classification" if is_classification else "Regression"
        missing_target = int(target_series.isnull().sum())

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{target_type}</div><div class="metric-lbl">DATA TYPE</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{unique_vals}</div><div class="metric-lbl">UNIQUE VALUES</div></div>', unsafe_allow_html=True)
        with m3:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{task_type}</div><div class="metric-lbl">DETECTED TASK</div></div>', unsafe_allow_html=True)
        with m4:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{missing_target}</div><div class="metric-lbl">MISSING TARGET ROWS</div></div>', unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("📊 Target Variable Visualization")
        
        chart_type = st.radio(
            "Select Chart Type:",
            ["Bar Chart", "Pie Chart", "Histogram / Distribution", "Box Plot", "Line / Sequence"],
            horizontal=True
        )

        fig, ax = plt.subplots(figsize=(9, 3.8))
        fig.patch.set_facecolor('#0f1117')
        ax.set_facecolor('#0f1117')

        if chart_type == "Bar Chart":
            target_series.value_counts().head(10).plot(kind='bar', color='#d4af37', ax=ax)
        elif chart_type == "Pie Chart":
            target_series.value_counts().head(8).plot(kind='pie', autopct='%1.1f%%', colors=['#d4af37', '#e5c158', '#8a8f9d', '#4a4f5d'], ax=ax)
        elif chart_type == "Histogram / Distribution":
            sns.histplot(target_series.dropna(), kde=True, color='#d4af37', ax=ax)
        elif chart_type == "Box Plot":
            sns.boxplot(x=target_series.dropna(), color='#d4af37', ax=ax)
        elif chart_type == "Line / Sequence":
            ax.plot(target_series.values, color='#d4af37', lw=1.8)

        ax.tick_params(colors='#e0e0e0')
        for spine in ax.spines.values():
            spine.set_color('#232733')
        st.pyplot(fig)

        st.markdown("---")
        if st.button("🚀 Launch AutoML Training Pipeline"):
            temp_csv = "data/active_temp.csv"
            active_df.to_csv(temp_csv, index=False)
            
            with st.spinner(f"🤖 Training model on target '{target_col}', executing pipeline, and generating report..."):
                res = run_autonomous_ml_pipeline(
                    csv_path=temp_csv,
                    target_col=target_col,
                    selected_model="Random Forest",
                    preprocessing_steps=["Median Imputation", "Text/Categorical Encoding"]
                )
                st.session_state.agent_result = res

        if st.session_state.agent_result:
            res = st.session_state.agent_result
            st.markdown("---")
            
            if res["success"]:
                st.markdown("""
                <div class="success-card">
                    <h2 style="margin: 0; color: #d4af37;">⚡ Trained Pipeline & Model Ready</h2>
                    <p style="color: #a0a5b5; margin: 5px 0 0 0; font-weight: 600;">AUTONOMOUS ML PIPELINE COMPLETED SUCCESSFULLY</p>
                </div>
                """, unsafe_allow_html=True)

                col_d1, col_d2 = st.columns(2)
                model_pkl = "data/best_model.pkl"
                with col_d1:
                    if os.path.exists(model_pkl):
                        with open(model_pkl, "rb") as f:
                            st.download_button(
                                label="📥 Download Trained Model (.pkl)",
                                data=f.read(),
                                file_name=f"{target_col}_model.pkl",
                                mime="application/octet-stream"
                            )

                ml_report_text = f"""# Final AutoML Benchmark Report: {target_col} Modeling

Target Variable: {target_col} | Task Type: {task_type}
Records Evaluated: {active_df.shape[0]} | Features Count: {active_df.shape[1]}

## 1. Model Training & Performance Metrics
{res['output']}

## 2. Artifact & Deployment
- Serialized Model: {target_col}_model.pkl
- Preprocessing Included: TF-IDF / One-Hot Encoding + Median Imputation
- Engine: AutoData Autonomous Agent (Gemini Flash)
"""
                with col_d2:
                    st.download_button(
                        label="📄 Download ML Executive Report (.txt)",
                        data=ml_report_text,
                        file_name=f"{target_col}_ml_executive_report.txt",
                        mime="text/plain"
                    )

                st.markdown("### 📑 Final ML Executive Report & Evaluation Summary")
                st.markdown(f'<div class="report-card">{ml_report_text}</div>', unsafe_allow_html=True)

                with st.expander("💻 View Generated Sandbox Python Code"):
                    st.code(res["code"], language="python")
            else:
                st.error("Pipeline failed to execute.")
                st.text_area("Error Log", res["output"], height=320)