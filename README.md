# Thoughtful AI – Support Agent

This small project provides a Streamlit-based FAQ assistant backed by a small, hard-coded dataset. It falls back to a CLI if Streamlit or scikit-learn are not installed.

Setup
1. Create a virtual environment (recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

Run

Web UI (Streamlit):

```bash
streamlit run app.py
```

Command-line fallback (no Streamlit required):

```bash
python3 app.py
```

If you want to enable the OpenAI fallback, set `OPENAI_API_KEY` in your environment before running.
