Autonomous EDA Agent
An intelligent multi-step workflow for automated exploratory data analysis.

🏗️ System Architecture

The high-level flow of the Autonomous EDA Agent follows a modular design so that additional analysis tools and checks can be added later:

1. Upload Dataset: Ingest CSV or Excel files through the web interface.
2. Data Profiling: Automatically extract shape, column types, and missingness.
3. Agent Decision: The agent selects the next useful EDA action dynamically based on intermediate results.
4. EDA Tools: Run specific analysis, statistics, and visualization tools.
5. Insights & Report: Present actionable observations and downloadable reports.

⚡ Technology Stack

* Python: Core application and analysis logic.
* Pandas & NumPy: Tabular data loading, manipulation, and numerical operations.
* Streamlit: Interactive web UI and dashboard.
* LangGraph & LangChain: Multi-step agent workflow orchestration, state maintenance, and tool integration.

🚀 Getting Started Locally

1. Clone & Set Up Environment

```bash
git clone <repository-url>
cd agentic-ml-pipeline
python -m venv venv
venv\Scripts\activate  # On Windows
# source venv/bin/activate  # On macOS/Linux
```

2. Install Dependencies

```bash
pip install -r requirements.txt
```

3. Configure API Keys

Create a `.env` file in the root directory and add your API credentials:

```
GEMINI_API_KEY=your_gemini_api_key_here
```

4. Run the Streamlit Application

```bash
streamlit run app.py
```

📋 Core Capabilities

* Automated Profiling: Inspect dataset dimensions, column types, missing values, and duplicate rows.
* Robust Statistics: Calculate means, medians, standard deviations, ranges, and record counts.
* Sandbox Execution: Securely run generated code with automatic error debugging.
* Rich Visualizations: Dynamically generate plots to support interpretation.
