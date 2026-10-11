"""How well does the matcher work? Results from evaluation/evaluate.py."""
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

REPORTS = Path(__file__).resolve().parent.parent / "reports"
BEFORE, AFTER, BAD = "#8b949e", "#00d2ff", "#ff6b6b"
LAYOUT = dict(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
              font=dict(color="#c9d1d9"), margin=dict(l=10, r=10, t=30, b=10))

st.set_page_config(page_title="Model Evaluation", page_icon="📈", layout="wide")


@st.cache_data
def load(name):
    return pd.read_csv(REPORTS / f"{name}.csv")


summary = load("eval_summary").set_index("version")
dataset = load("eval_dataset").iloc[0]
by_cat = load("eval_by_category")
fp = load("eval_skill_false_positives")
skills = load("skills_by_role")
orig, fixed = summary.loc["Original"], summary.loc["Fixed"]

st.title("📈 How well does the matcher work?")
st.markdown(
    f"Tested on **{int(dataset.unique_resumes)} real resumes** labelled with **{int(dataset.categories)} job "
    "categories**. Each resume is used as the job description in turn, and a good matcher ranks resumes "
    "from the **same category** first."
)

c1, c2, c3 = st.columns(3)
c1.metric("Right resume ranked #1", f"{fixed.small_pool_top1_accuracy:.0%}",
          help="Pools of 5 resumes (1 from the right category), like a recruiter's upload")
c2.metric("Unrelated resume scored over 50%", f"{fixed.single_upload_unrelated_scored_over_50pct:.0%}",
          delta=f"was {orig.single_upload_unrelated_scored_over_50pct:.0%}", delta_color="inverse",
          help="Uploading one resume from a different field")
c3.metric("Duplicate rows removed", f"{int(dataset.rows - dataset.unique_resumes)} of {int(dataset.rows)}",
          help="The public dataset repeats the same resumes many times")

st.markdown("### Before vs after the fixes")
metrics = {
    "Right resume #1 (pool of 5)": "small_pool_top1_accuracy",
    "Precision@5 (all 166)": "precision_at_5",
    "Unrelated resume scored >50%": "single_upload_unrelated_scored_over_50pct",
}
fig = go.Figure()
for version, color in [("Original", BEFORE), ("Fixed", AFTER)]:
    vals = [summary.loc[version, col] for col in metrics.values()]
    fig.add_bar(name=version, x=list(metrics), y=vals, marker_color=color,
                text=[f"{v:.0%}" for v in vals], textposition="outside")
fig.update_layout(barmode="group", height=360, yaxis=dict(tickformat=".0%", range=[0, 1.1]),
                  legend=dict(orientation="h", y=1.12), **LAYOUT)
st.plotly_chart(fig, width="stretch")
st.caption(
    "Ranking quality is about the same. The fix is in what the score means: before, a single unrelated "
    "resume usually showed as a strong match because the formula ignored every word the two documents "
    "didn't share."
)

left, right = st.columns(2)
with left:
    st.markdown("### Which roles it handles well")
    d = by_cat.sort_values("precision_at_5")
    colors = [AFTER if v >= 0.5 else BAD for v in d.precision_at_5]
    fig = go.Figure(go.Bar(x=d.precision_at_5, y=d.category, orientation="h", marker_color=colors,
                           text=[f"{v:.0%}" for v in d.precision_at_5], textposition="outside"))
    fig.add_vline(x=orig.random_baseline, line_dash="dot", line_color=BEFORE,
                  annotation_text="random", annotation_position="top")
    fig.update_layout(height=640, xaxis=dict(tickformat=".0%", range=[0, 1.15], title="Precision@5"), **LAYOUT)
    st.plotly_chart(fig, width="stretch")
    st.caption("Distinct vocabularies (Hadoop, Civil, Database) match well. Roles that share words with "
               "others (DevOps vs Testing vs Web) get confused: keyword matching can't tell them apart.")

with right:
    st.markdown("### Skill tagging: false matches")
    d = fp.sort_values("original_pct")
    fig = go.Figure()
    fig.add_bar(name="Original", y=d.skill, x=d.original_pct / 100, orientation="h", marker_color=BEFORE)
    fig.add_bar(name="Fixed", y=d.skill, x=d.fixed_pct / 100, orientation="h", marker_color=AFTER)
    fig.update_layout(barmode="group", height=360, xaxis=dict(tickformat=".0%", title="Resumes tagged"),
                      legend=dict(orientation="h", y=1.12), **LAYOUT)
    st.plotly_chart(fig, width="stretch")
    st.caption("The original tagged every resume with R and most with Go, because it matched letters "
               "inside other words. \"less\" is still over-tagged: it's also an everyday word.")

    st.markdown("### Top skills by role")
    role = st.selectbox("Role", sorted(skills.category.unique()),
                        index=sorted(skills.category.unique()).index("Data Science"))
    d = skills[skills.category == role].sort_values("pct_of_resumes")
    fig = go.Figure(go.Bar(x=d.pct_of_resumes / 100, y=d.skill, orientation="h", marker_color=AFTER,
                           text=[f"{v}%" for v in d.pct_of_resumes], textposition="outside"))
    fig.update_layout(height=240, xaxis=dict(tickformat=".0%", range=[0, 1.15],
                                             title=f"Share of {role} resumes"), **LAYOUT)
    st.plotly_chart(fig, width="stretch")

with st.expander("Method and limitations"):
    st.markdown(f"""
- **Data:** Kaggle "Resume Dataset" ({int(dataset.rows)} rows). Only {int(dataset.unique_resumes)} are unique, so
  results come from a small sample: treat differences of a few points as noise.
- **Test:** each resume is the query; it's compared with the others using the app's own pipeline.
  Category labels stand in for "relevant", which is a simplification: a Python Developer resume can be a
  good fit for a Data Science job.
- **Random baseline:** picking resumes at random would get precision@5 of {orig.random_baseline:.0%}.
- Reproduce with `python -m evaluation.evaluate`.
""")
