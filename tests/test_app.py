"""Smoke tests: both dashboard pages run end to end without errors."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent


def test_matching_page_runs_on_sample_data():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    for box in app.checkbox[:2]:  # sample JD + sample resumes
        box.check()
    app.run()
    next(b for b in app.button if "Matching" in b.label).click().run()
    app.run()
    assert not app.exception
    assert len(app.tabs) == 4


def test_evaluation_page_runs():
    page = AppTest.from_file(str(ROOT / "pages" / "1_📈_Model_Evaluation.py"), default_timeout=60).run()
    assert not page.exception
    assert len(page.metric) == 3
