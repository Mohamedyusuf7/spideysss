=====================================================
Autonomous EDA & ML Intelligence Agent
An intelligent, multi-step agentic workflow for automated exploratory data analysis (EDA) and self-correcting machine learning pipeline generation.
=====================================================

=====================================================
 System Architecture
The application follows a modular, stateful agent architecture powered by LangGraph and Gemini 3.5 Flash, cleanly separating reasoning from execution.

[ Upload Dataset ] ➔ [ Data Profiling ] ➔ [ Agent Planning (Think) ] 
                                                      │
[ Executive Report & UI ] ⟵ [ Sandbox Execution ] ⟵ [ Code Synthesis ]

1.Dataset Ingestion & Validation: Raw CSV or Excel files are uploaded and scanned for schema integrity, missing values, duplicates, and column metadata.  
2.Strategy Planner: The agent analyzes the dataset profile to formulate a structured multi-step preprocessing and modeling execution plan.  
3.Code Synthesizer: Generates complete, executable Python code tailored to the dataset while applying robust safety checks and data leakage prevention.
4.Secure Sandbox Execution: Executes the generated code locally, capturing standard output logs and generating dynamic EDA visualizations (eda_plot.png).
5.Executive Synthesis: Compiles a detailed markdown report summarizing cleaning steps, data patterns, and evaluation metrics. 
======================================================

======================================================
 Tech Stack
Orchestration: LangGraph (Stateful Multi-Agent Workflow)  
LLM Engine: Google Gemini 3.5 Flash (via LangChain)
User Interface: Streamlit (Enterprise Light & Green Theme)  
Data Processing & ML: Pandas, NumPy, Scikit-Learn  
Visualizations: Plotly & Matplotlib / Seaborn
======================================================

======================================================
 Getting Started Locally
 
1. Clone the Repository
Bash
git clone https://github.com/your-username/agentic-ml-pipeline.git
cd agentic-ml-pipeline

2. Create and Activate a Virtual Environment
Bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

3. Install Dependencies
Bash
pip install -r requirements.txt

5. Configure Environment Variables
Create a .env file in the root directory and add your Google Gemini API key:

Code snippet
GEMINI_API_KEY=your_actual_api_key_here
5. Run the Streamlit Application
Bash
streamlit run app.py
======================================================

======================================================
 FeaturesAutomated Data Profiling: 
 1.Instant statistical summaries, missingness percentages, duplicate metrics, and memory usage tracking.  
 2.Self-Correcting Execution: Automatic sandbox debugging that catches runtime exceptions and applies code patches.
 3.Interactive Dashboard: Tabbed interface offering an Executive Report, Visualizations Explorer, Generated Python Pipeline, and Terminal Logs.
 4.Export Capabilities: Download cleaned datasets, generated Python scripts, and comprehensive markdown audit reports directly. 
 ======================================================
