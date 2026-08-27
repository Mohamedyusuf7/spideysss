# Autonomous Agentic ML & EDA Intelligence Platform

An enterprise-grade, self-correcting agentic system built with LangGraph and Google Gemini for autonomous exploratory data analysis, data cleaning, and machine learning pipeline generation.

## 🏗️ System Architecture & Workflow

The platform utilizes a stateful multi-agent workflow that cleanly separates high-level reasoning from secure local computation.

```
+-------------------------------------------------------------------------+
|                        1. Dataset Ingestion                             |
|              (Raw CSV/Excel Upload & Schema Validation)                 |
+-------------------------------------------------------------------------+
                                     │
                                     ▼
+-------------------------------------------------------------------------+
|                        2. Data Inspector Node                           |
|        (Analyzes Rows, Columns, Data Types, and Missingness)            |
+-------------------------------------------------------------------------+
                                     │
                                     ▼
+-------------------------------------------------------------------------+
|                        3. Strategy Planner Node                         |
|     (Formulates Multi-Step Preprocessing & Modeling Execution Plan)     |
+-------------------------------------------------------------------------+
                                     │
                                     ▼
+-------------------------------------------------------------------------+
|                     4. Code Synthesizer (Generator)                     |
|         (Writes Executable Python Code with Leakage Prevention)         |
+-------------------------------------------------------------------------+
                                     │
                                     ▼
+-------------------------------------------------------------------------+
|                         5. Secure Sandbox Executor                      |
|       (Executes Code, Captures Standard Output, Saves EDA Artifacts)    |
+-------------------------------------------------------------------------+
                         │                               │
         [ If Runtime Error / Exception ]     [ If Execution Successful ]
                         │                               │
                         ▼                               ▼
        +-------------------------------+  +------------------------------+
        |     6. Self-Correction Loop   |  |     7. Executive Synthesizer |
        | (Feeds Error Back to Coder)   |  |    (Compiles Final Report)   |
        +-------------------------------+  +------------------------------+
```

## ⚡ Core Modules & Features

- **Module 01 — Validation & Health**: Automatically checks datasets for null rows, duplicate entries, constant zero-variance columns, and missing sentinel values, issuing a unified Data Quality Score.
- **Module 02 — Intelligence & Profiling**: Generates column metadata, distribution histograms, box plots, correlation heatmaps, and LLM-driven analytical insights.
- **Module 03 — Data Transformation**: Implements automated pipelines for dropping duplicates, cleaning null fields, imputing missing data (median/mode), and handling outlier removal via the IQR method.
- **Module 04 — AutoML Model Engine**: Automatically detects task types (Classification vs. Regression), applies preprocessing transforms, trains baseline models (Logistic/Linear Regression, Decision Trees, Random Forests), and outputs a performance leaderboard.
- **Self-Correcting Execution Sandbox**: Captures standard output (stdout) logs and automatically traps runtime exceptions to patch and re-run generated scripts dynamically.

## 🛠️ Technology Stack

- **Orchestration**: LangGraph (Stateful Agentic Graph Architecture)
- **Intelligence Layer**: Google Gemini 3.5 Flash (via LangChain)
- **User Interface**: Streamlit (Enterprise Dashboard Design)
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-Learn (Pipelines, Transformers, Ensembles)
- **Visualization**: Plotly & Seaborn

## 📦 Getting Started & Local Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/agentic-ml-pipeline.git
cd agentic-ml-pipeline
```

### 2. Create and Activate a Virtual Environment

```bash
python -m venv venv
# On Windows (PowerShell):
venv\Scripts\activate
# On macOS / Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the root directory of your project folder and add your Gemini API key:

```
GEMINI_API_KEY=your_actual_gemini_api_key_here
```

### 5. Launch the Application

```bash
streamlit run app.py
```

## 📋 Repository Structure

```
agentic-ml-pipeline/
│
├── agent/
│   ├── graph.py         # LangGraph state machine & node logic
│   └── state.py         # Agent state schema definition
│
├── data_store/          # Storage directory for uploads and audit reports
├── app.py               # Main Streamlit web dashboard application
├── requirements.txt     # Python project package dependencies
└── README.md            # Project documentation
```
