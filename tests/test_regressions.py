"""Regression tests for bugs found in the original matcher.

Each test failed before its fix, so the bug can't quietly come back.
"""
from src.similarity.ranker import extract_skills, rank_resumes

JD = "Data scientist with Python, machine learning, SQL and statistics experience."
GOOD = "Data scientist. Python, machine learning, SQL, statistics, pandas."
CHEF = "Chef with ten years cooking Italian food in restaurants. Python hobbyist."
OTHER = "Marketing manager running social media campaigns and brand events."


def score(resumes, name):
    return next(r["score_pct"] for r in rank_resumes(JD, resumes) if r["name"] == name)


def test_single_unrelated_resume_is_not_a_perfect_match():
    # Before: IDF was 0 for every word not shared by JD and resume, so one
    # shared word ("python") made the chef a 100% match.
    assert score({"chef": CHEF}, "chef") < 20


def test_score_does_not_depend_on_other_uploaded_resumes():
    # Before: the chef scored 25.9% next to one resume and 0% next to two.
    alone = score({"chef": CHEF}, "chef")
    with_others = score({"good": GOOD, "chef": CHEF, "other": OTHER}, "chef")
    assert abs(alone - with_others) < 5


def test_relevant_resume_ranks_first_and_scores_high():
    ranked = rank_resumes(JD, {"chef": CHEF, "good": GOOD, "other": OTHER})
    assert ranked[0]["name"] == "good"
    assert ranked[0]["score_pct"] > 70


def test_skills_match_whole_words_only():
    # Before: substring matching found java/go/r in this sentence.
    found = extract_skills("I enjoy reading. Worked at Google on javascript. R&D team.")["all_skills"]
    assert found == ["javascript"]


def test_skills_with_symbols_and_short_names_still_match():
    found = set(extract_skills("Skills: C++, C#, .NET, Node.js, R, Go, SQL Server")["all_skills"])
    assert {"c++", "c#", ".net", "node.js", "r", "go", "sql server"} <= found
