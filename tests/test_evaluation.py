"""The evaluation metrics behave as expected on tiny, obvious data."""
from evaluation.evaluate import evaluate_ranking, old_extract_skills, old_idf
from src.similarity.ranker import load_skills_database, preprocess
from src.vectorization.idf import compute_idf

TEXTS = [
    ("data", "python pandas statistics machine learning models"),
    ("data", "python statistics machine learning pandas numpy"),
    ("chef", "cooking kitchen menu restaurant recipes"),
    ("chef", "kitchen cooking recipes menu catering"),
]


def test_ranking_eval_on_obvious_categories():
    docs = [preprocess(t) for _, t in TEXTS]
    labels = [c for c, _ in TEXTS]
    result = evaluate_ranking(docs, labels, compute_idf, k=1)
    assert result["top1_correct"].all()
    assert result["precision_at_k"].mean() == 1.0


def test_old_idf_is_kept_as_the_original_formula():
    idf = old_idf([["a", "b"], ["a"]])
    assert idf["a"] < 0  # log(2/3): the original went negative


def test_old_skill_extractor_reproduces_the_substring_bug():
    assert "java" in old_extract_skills("javascript developer", load_skills_database())
