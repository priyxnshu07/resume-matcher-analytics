"""How well does the matcher work on real resumes?

    python -m evaluation.evaluate

Uses a public dataset of resumes labelled with a job category (Data Science,
HR, Java Developer, ...). Each resume is used in turn as the "job
description" and the matcher ranks all the others. A good matcher puts
resumes from the same category at the top.

Metrics
- Top-1 accuracy: is the best match from the same category?
- Precision@5: share of the top 5 matches from the same category.

Writes CSVs to reports/ for the dashboard's "Evaluation" tab.
"""
import math
import re
import urllib.request
from collections import Counter
from pathlib import Path

import pandas as pd

from src.similarity.cosine import cosine_similarity
from src.similarity.ranker import extract_skills, load_skills_database, preprocess
from src.vectorization.idf import compute_idf
from src.vectorization.tfidf import compute_tfidf

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "eval" / "resumes.csv"
REPORTS = ROOT / "reports"

# Kaggle "Resume Dataset" (UpdatedResumeDataSet.csv), from a GitHub copy
# pinned to one commit so the numbers are reproducible.
DATA_URL = ("https://raw.githubusercontent.com/Priyanshu-1729/Resume-Screening-using-Python/"
            "ea8ad5e877a2258154fdc65660875869305aaa4b/UpdatedResumeDataSet.csv")
K = 5


# ---------- the original (buggy) versions, kept only for comparison ----------

def old_idf(documents):
    n = len(documents)
    df = Counter(word for doc in documents for word in set(doc))
    return {w: math.log(n / (1 + c)) for w, c in df.items()}


def old_extract_skills(text, skills_db):
    text = text.lower()
    return sorted({s for skills in skills_db.values() for s in skills if s.lower() in text})


# ---------- data ----------

def load_resumes():
    if not DATA.exists():
        DATA.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(DATA_URL, DATA)
    raw = pd.read_csv(DATA)
    # The dataset repeats the same resumes many times; keep each one once.
    unique = raw.drop_duplicates("Resume").reset_index(drop=True)
    # Strip non-ASCII artefacts left over from PDF conversion.
    unique["Resume"] = unique["Resume"].map(lambda t: re.sub(r"[^\x00-\x7f]", " ", t))
    return raw, unique


# ---------- evaluation ----------

def evaluate_ranking(docs, labels, idf_fn, k=K):
    """Leave-one-out: each resume is the query, all others are ranked."""
    idf = idf_fn(docs)
    vectors = [compute_tfidf(d, idf) for d in docs]
    rows = []
    for i, query in enumerate(vectors):
        scores = [(cosine_similarity(query, v), j) for j, v in enumerate(vectors) if j != i]
        scores.sort(key=lambda s: (-s[0], s[1]))
        top = [labels[j] for _, j in scores[:k]]
        rows.append({
            "category": labels[i],
            "top1_correct": top[0] == labels[i],
            "precision_at_k": sum(t == labels[i] for t in top) / k,
            "top1_score": scores[0][0],
        })
    return pd.DataFrame(rows)


def evaluate_small_pools(docs, labels, idf_fn, pool_size=5, pools_per_query=3, seed=7):
    """The app's real use: one job, a handful of uploaded resumes.

    Each pool has 1 resume from the query's category and pool_size - 1 from
    other categories. IDF is computed over the query + pool only, exactly as
    rank_resumes() does. Also scores single-resume uploads of an unrelated
    resume, where the original formula could report a perfect match.
    """
    import random
    rng = random.Random(seed)
    by_label = {}
    for i, lab in enumerate(labels):
        by_label.setdefault(lab, []).append(i)

    def scores_for(query, pool):
        idf = idf_fn([docs[query]] + [docs[j] for j in pool])
        q = compute_tfidf(docs[query], idf)
        return [cosine_similarity(q, compute_tfidf(docs[j], idf)) for j in pool]

    top1, unrelated_scores, single_scores = [], [], []
    for i, lab in enumerate(labels):
        same = [j for j in by_label[lab] if j != i]
        others = [j for j in range(len(labels)) if labels[j] != lab]
        if not same:
            continue
        for _ in range(pools_per_query):
            pool = [rng.choice(same)] + rng.sample(others, pool_size - 1)
            s = scores_for(i, pool)
            top1.append(max(range(pool_size), key=lambda x: s[x]) == 0)
            unrelated_scores.extend(s[1:])
        single_scores.append(scores_for(i, [rng.choice(others)])[0])

    return {
        "small_pool_top1_accuracy": sum(top1) / len(top1),
        "avg_score_unrelated_resume": sum(unrelated_scores) / len(unrelated_scores),
        "single_upload_unrelated_scored_over_50pct": sum(x > 0.5 for x in single_scores) / len(single_scores),
    }


def main():
    REPORTS.mkdir(exist_ok=True)
    raw, df = load_resumes()
    labels = df["Category"].tolist()
    docs = [preprocess(t) for t in df["Resume"]]
    skills_db = load_skills_database()

    # 1. Ranking quality, before vs after the IDF fix
    before = evaluate_ranking(docs, labels, old_idf)
    after = evaluate_ranking(docs, labels, compute_idf)
    summary = pd.DataFrame([
        {"version": "Original", "top1_accuracy": before.top1_correct.mean(),
         "precision_at_5": before.precision_at_k.mean(), **evaluate_small_pools(docs, labels, old_idf)},
        {"version": "Fixed", "top1_accuracy": after.top1_correct.mean(),
         "precision_at_5": after.precision_at_k.mean(), **evaluate_small_pools(docs, labels, compute_idf)},
    ])
    # Random baseline: chance a random other resume shares the category
    counts = Counter(labels)
    chance = sum(c * (c - 1) for c in counts.values()) / (len(labels) * (len(labels) - 1))
    summary["random_baseline"] = chance
    summary = summary.round(3)
    summary.to_csv(REPORTS / "eval_summary.csv", index=False)

    # 2. Which job categories the matcher handles well or badly
    by_cat = (after.groupby("category")
              .agg(resumes=("category", "size"),
                   precision_at_5=("precision_at_k", "mean"),
                   top1_accuracy=("top1_correct", "mean"))
              .round(3).sort_values("precision_at_5", ascending=False).reset_index())
    by_cat.to_csv(REPORTS / "eval_by_category.csv", index=False)

    # 3. Skill extraction: how often each method tags common false positives
    n = len(df)
    old_sets = [set(old_extract_skills(t, skills_db)) for t in df["Resume"]]
    new_sets = [set(extract_skills(t, skills_db)["all_skills"]) for t in df["Resume"]]
    fp = pd.DataFrame([
        {"skill": s,
         "original_pct": round(100 * sum(s in x for x in old_sets) / n, 1),
         "fixed_pct": round(100 * sum(s in x for x in new_sets) / n, 1)}
        for s in ["r", "go", "java", "c#", "rest", "less", "express"]
    ])
    fp.to_csv(REPORTS / "eval_skill_false_positives.csv", index=False)
    avg_skills = pd.DataFrame([{
        "original_avg_skills": round(sum(map(len, old_sets)) / n, 1),
        "fixed_avg_skills": round(sum(map(len, new_sets)) / n, 1),
    }])
    avg_skills.to_csv(REPORTS / "eval_skill_counts.csv", index=False)

    # 4. Skills by role: the most common skills in each category's resumes
    rows = []
    for cat, group in df.assign(skills=new_sets).groupby("Category"):
        tally = Counter(s for skills in group["skills"] for s in skills)
        for skill, c in tally.most_common(5):
            rows.append({"category": cat, "skill": skill, "resumes": len(group),
                         "pct_of_resumes": round(100 * c / len(group))})
    pd.DataFrame(rows).to_csv(REPORTS / "skills_by_role.csv", index=False)

    dataset = pd.DataFrame([{"rows": len(raw), "unique_resumes": n, "categories": len(counts)}])
    dataset.to_csv(REPORTS / "eval_dataset.csv", index=False)

    print(dataset.to_string(index=False), "\n")
    print(summary.T.to_string(), "\n")
    print(by_cat.to_string(index=False), "\n")
    print(fp.to_string(index=False))
    print(avg_skills.to_string(index=False))


if __name__ == "__main__":
    main()
