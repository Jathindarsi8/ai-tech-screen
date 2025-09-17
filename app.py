# app.py
import os
import json


def load_questions():
    """Try to load questions from data.json; fallback to data.py if missing/invalid."""
    json_path = os.path.join(os.path.dirname(__file__), "data.json")
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                obj = json.load(f)
                if isinstance(obj, dict) and "questions" in obj:
                    return obj["questions"]
        except Exception:
            # fall through to python module fallback
            pass
    # fallback to python module
    try:
        from data import QUESTIONS
        return QUESTIONS
    except Exception:
        return []


QUESTIONS = load_questions()

# Optional dependencies (we handle missing packages gracefully so the repo is
# usable even if you haven't installed everything yet).
try:
    import streamlit as st
except Exception:
    st = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:
    TfidfVectorizer = None
    cosine_similarity = None

# --- Optional LLM fallback (OpenAI). ---
OPENAI_ENABLED = False
try:
    from openai import OpenAI
    if os.environ.get("OPENAI_API_KEY"):
        client = OpenAI()
        OPENAI_ENABLED = True
except Exception:
    OPENAI_ENABLED = False


def build_vector_index():
    """Build TF-IDF index if sklearn is available; otherwise return None."""
    if TfidfVectorizer is None:
        return None
    corpus = [q["question"] + " " + q["answer"] for q in QUESTIONS]
    vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), stop_words="english")
    X = vectorizer.fit_transform(corpus)
    return (vectorizer, X)


VECTOR_INDEX = build_vector_index()


def retrieve_best(query: str):
    """Return best match and score. If sklearn isn't available, use simple substring matching."""
    if not query or not query.strip():
        return None, 0.0

    if VECTOR_INDEX is not None:
        vectorizer, X = VECTOR_INDEX
        q_vec = vectorizer.transform([query])
        sims = cosine_similarity(q_vec, X)[0]
        best_idx = int(sims.argmax())
        best_score = float(sims[best_idx])
        return QUESTIONS[best_idx], best_score

    # Fallback: token overlap (Jaccard) between query tokens and each QA text.
    # This handles short queries better than substring matching.
    def tokens(s: str):
        return set([t for t in ''.join(c if c.isalnum() else ' ' for c in s.lower()).split() if t])

    q_tokens = tokens(query)
    best_idx = None
    best_score = 0.0
    for i, item in enumerate(QUESTIONS):
        hay_tokens = tokens(item["question"] + " " + item["answer"])
        if not hay_tokens:
            continue
        inter = len(q_tokens & hay_tokens)
        union = len(q_tokens | hay_tokens)
        score = float(inter) / float(union) if union else 0.0
        if score > best_score:
            best_score = score
            best_idx = i

    if best_idx is not None and best_score > 0.0:
        return QUESTIONS[best_idx], best_score
    return None, 0.0


def fallback_llm(query: str) -> str:
    """Use OpenAI if available; otherwise return a safe, generic message."""
    if OPENAI_ENABLED:
        msg = [
            {"role": "system", "content": "You are a helpful, concise support agent for Thoughtful AI."},
            {"role": "user", "content": query},
        ]
        resp = client.chat.completions.create(model="gpt-4o-mini", messages=msg, temperature=0.2)
        return resp.choices[0].message.content.strip()
    return (
        "I couldn\'t find this in our predefined FAQ. Please rephrase your question, "
        "or ask about EVA, CAM, PHIL, or the benefits of Thoughtful AI\'s agents."
    )


CONFIDENCE_THRESHOLD = 0.28  # tuned to be a bit forgiving for short queries


def run_streamlit():
    """Run the Streamlit UI. Assumes `st` is available."""
    st.set_page_config(page_title="Thoughtful AI – Support Agent", page_icon="🤖")
    st.title("🤖 Thoughtful AI – Support Agent")
    st.caption("Answers common questions from a predefined set. Falls back to an LLM for everything else.")

    if "history" not in st.session_state:
        st.session_state.history = []

    with st.chat_message("assistant"):
        st.markdown("Hi! Ask me about Thoughtful AI's agents (EVA, CAM, PHIL) or their benefits.")

    user_query = st.chat_input("Type your question…")

    if user_query:
        # Show user message
        with st.chat_message("user"):
            st.markdown(user_query)

        best_item, score = retrieve_best(user_query)

        if best_item and score >= CONFIDENCE_THRESHOLD:
            answer = best_item["answer"]
            source = f"**(matched FAQ: “{best_item['question']}”, confidence={score:.2f})**"
        else:
            answer = fallback_llm(user_query)
            source = "**(fallback)**"

        with st.chat_message("assistant"):
            st.markdown(answer)
            st.caption(source)

        st.session_state.history.append({"q": user_query, "a": answer, "meta": {"score": score, "source": source}})


def run_cli():
    """Simple command-line fallback so the project is usable without Streamlit or sklearn."""
    print("Thoughtful AI – Support Agent (CLI fallback)")
    print("Type 'quit' to exit.\n")
    while True:
        try:
            q = input("Question: ")
        except EOFError:
            break
        if not q:
            continue
        if q.strip().lower() in ("quit", "exit"):
            break
        best_item, score = retrieve_best(q)
        if best_item and score >= CONFIDENCE_THRESHOLD:
            print('\nAnswer:', best_item["answer"])
            print(f"(matched FAQ: {best_item['question']}, confidence={score:.2f})\n")
        else:
            print('\nAnswer:', fallback_llm(q))
            print("(fallback)\n")


def main():
    if st is not None:
        run_streamlit()
    else:
        print("Streamlit is not installed or not available in this environment.")
        print("To run the app with the web UI, install dependencies: pip install -r requirements.txt\n")
        run_cli()


if __name__ == "__main__":
    main()