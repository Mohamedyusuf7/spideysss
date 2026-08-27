import os
import time
from dotenv import load_dotenv
from google import genai
from tools import execute_python_code

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Initialize Gemini Client
client = genai.Client(api_key=api_key)

def call_gemini_with_fallback(prompt: str, max_retries: int = 4):
    """
    Calls verified active Gemini endpoints with exponential backoff on transient spikes.
    """
    models_to_try = [
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3-flash-preview",
    "gemini-2.5-pro",
    "gemini-2.5-flash-lite"
]
    
    last_exception = None
    for model_name in models_to_try:
        for attempt in range(1, max_retries + 1):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )
                return response.text, model_name
            except Exception as e:
                last_exception = e
                err_str = str(e)
                if "503" in err_str or "429" in err_str or "UNAVAILABLE" in err_str:
                    time.sleep(2 * attempt)
                else:
                    break
                    
    raise last_exception

def generate_dataset_intelligence_report(df_summary: str, sample_head: str) -> str:
    """
    Generates an exhaustive, highly detailed markdown report with a big dynamic title.
    """
    prompt = f"""You are a Lead AI Data Scientist and Senior Data Auditor.
Write an exhaustive, deeply analytical, and professional markdown report.
Do not provide a generic summary; analyze the dataset's operational domain, data health, possible data leakage risks, feature interactions, and production readiness.

Dataset Metadata & Summary:
{df_summary}

Dataset Preview (First Rows):
{sample_head}

CRITICAL FORMATTING INSTRUCTIONS:
- On the very first line, output ONLY a big, intelligent, domain-specific Title starting with '# ' (Example: # Comprehensive Data Audit & Predictive Integrity Report: RMS Titanic Passenger Survival Architecture OR # Home Valuation & Rental Yield Intelligence Report).
- Do NOT output generic filenames or raw CSV names in the title.
- Organize the body into clear numbered sections:
  ## 1. Executive Summary
  ### 1.1 Scope and Domain Analysis
  ### 1.2 Top-Level Dataset Health Metrics
  ## 2. Data Quality & Imputation Methods
  ## 3. Structural Patterns & Data Leakage Audit
  ## 4. Recommended Machine Learning Roadmap
  ## 5. Production Recommendations & Risk Analysis
- Provide thorough, multi-paragraph analysis under each section.
"""
    content, _ = call_gemini_with_fallback(prompt)
    clean_report = str(content)
    if "extras': {'signature':" in clean_report:
        clean_report = clean_report.split("extras': {'signature':")[0].rstrip("', \n")
    return clean_report.strip()

def run_autonomous_ml_pipeline(
    csv_path: str,
    target_col: str,
    selected_model: str,
    preprocessing_steps: list,
    max_retries: int = 3
):
    trace_logs = []
    prep_instructions = ", ".join(preprocessing_steps) if preprocessing_steps else "Standard automated cleaning"
    
    current_prompt = (
        f"You are an expert Autonomous Data Scientist Agent.\n"
        f"Dataset path: '{csv_path}'\n"
        f"Target Column: '{target_col}'\n"
        f"Desired Model Family: '{selected_model}'\n"
        f"User Preprocessing Choices: {prep_instructions}\n\n"
        f"Write complete, executable Python code that performs:\n"
        f"1. COMPREHENSIVE AUTO-EDA & MULTI-PLOT DASHBOARD:\n"
        f"   - Inspect and print dataset shape, info, and missing value counts.\n"
        f"   - Create a clean 2x2 multi-subplot figure using matplotlib and seaborn:\n"
        f"       * Subplot 1: Target Variable ('{target_col}') Distribution.\n"
        f"       * Subplot 2: Numeric Feature Correlation Heatmap.\n"
        f"       * Subplot 3: Missing Value Counts per column bar chart.\n"
        f"       * Subplot 4: Distribution or Boxplot of key predictive features against target.\n"
        f"   - Save the figure directly to 'data/eda_chart.png' (plt.tight_layout(); plt.savefig('data/eda_chart.png', dpi=200, bbox_inches='tight'); plt.close()).\n"
        f"2. DATA PREPROCESSING & TEXT VECTORIZATION:\n"
        f"   - If text features exist (e.g. message text), use TfidfVectorizer or CountVectorizer.\n"
        f"   - Handle nulls, encode categoricals, drop unnecessary IDs/names, and scale numeric features if requested.\n"
        f"3. MODEL TRAINING & EVALUATION:\n"
        f"   - Split into 80% train and 20% test sets.\n"
        f"   - Train a '{selected_model}' (use appropriate scikit-learn classifier or regressor depending on target type).\n"
        f"   - Print performance metrics (Accuracy, Precision, Recall, F1, Confusion Matrix or RMSE/R2).\n"
        f"4. MODEL EXPORT:\n"
        f"   - Import joblib and serialize the fitted model pipeline to 'data/best_model.pkl' (import joblib; joblib.dump(model, 'data/best_model.pkl')).\n"
        f"5. FINAL EXECUTIVE REPORT:\n"
        f"   - Print '================= FINAL EXECUTIVE REPORT ================='\n"
        f"   - Detail: 1) Data Profile 2) Preprocessing Applied 3) Target Column & Model Results.\n\n"
        f"IMPORTANT RULES:\n"
        f"- Flat, top-level code only. DO NOT wrap inside 'def main()' or 'if __name__ == \"__main__\"'.\n"
        f"- Use print() statements generously.\n"
        f"- Return ONLY executable Python code inside a markdown python block."
    )

    for attempt in range(1, max_retries + 1):
        trace_logs.append(f"🔄 **Attempt {attempt} of {max_retries}:** Generating ML Pipeline...")
        try:
            generated_code, used_model = call_gemini_with_fallback(current_prompt)
            trace_logs.append(f"⚡ Connected to `{used_model}` successfully.")
        except Exception as e:
            trace_logs.append(f"❌ **API Error:** {str(e)}")
            return {
                "success": False,
                "code": "",
                "output": f"API Error: {str(e)}",
                "logs": trace_logs
            }

        trace_logs.append("💻 **Executing code in Python sandbox...**")
        exec_result = execute_python_code(generated_code)
        
        if exec_result["status"] == "SUCCESS":
            trace_logs.append("✅ **Success!** Code executed without any runtime errors.")
            return {
                "success": True,
                "code": generated_code,
                "output": exec_result["output"],
                "logs": trace_logs
            }
        else:
            error_msg = exec_result["error"]
            trace_logs.append(f"⚠️ **Runtime Error:** `{error_msg}`")
            trace_logs.append("🧠 **Self-Correction:** Sending traceback to Gemini to patch and regenerate...")
            
            current_prompt = (
                "The previous code failed with a runtime error.\n\n"
                f"Failed Code:\n{generated_code}\n\n"
                f"Runtime Error Message:\n{error_msg}\n\n"
                "Please fix the bug and return ONLY flat, immediately executable Python code inside a markdown python code block."
            )

    trace_logs.append("❌ **Failed:** Maximum retries reached.")
    return {
        "success": False,
        "code": generated_code,
        "output": exec_result["error"],
        "logs": trace_logs
    }
