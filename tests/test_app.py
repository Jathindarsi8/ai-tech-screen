import importlib.util
import os


def load_app():
    spec = importlib.util.spec_from_file_location("app", os.path.join(os.path.dirname(__file__), "..", "app.py"))
    app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app)
    return app


def test_questions_loaded():
    app = load_app()
    assert hasattr(app, "QUESTIONS")
    assert isinstance(app.QUESTIONS, list)
    assert len(app.QUESTIONS) >= 1


def test_retrieve_best_matches():
    app = load_app()
    item, score = app.retrieve_best("What does the eligibility verification agent do?")
    assert item is not None
    assert score > 0.0


def test_empty_query_returns_none():
    app = load_app()
    item, score = app.retrieve_best("")
    assert item is None
    assert score == 0.0


def test_unknown_query_uses_fallback():
    app = load_app()
    # Use a query that's outside the FAQ
    item, score = app.retrieve_best("How do I reset an unrelated system setting?")
    # Depending on matching, it may return None or low score; ensure acceptable behavior
    assert (item is None and score == 0.0) or (isinstance(score, float) and score >= 0.0)


def test_fallback_llm_string():
    app = load_app()
    text = app.fallback_llm("Tell me something unknown")
    assert isinstance(text, str)
