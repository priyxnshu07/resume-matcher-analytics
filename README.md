# NLP‑based Resume Parser & Matcher

> Rank resumes against a job description using **TF‑IDF** and **Cosine Similarity** — every core algorithm written from scratch in Python.

🔗 **Live Demo:** [nlp-based-resume-matching.streamlit.app](https://nlp-based-resume-matching.streamlit.app/)

---

## Overview

Recruiters paste a job description, upload resumes, and instantly see candidates ranked by relevance. The system extracts skills, highlights gaps, and explains *why* each score was assigned — all through a Streamlit dashboard.

### Core Pipeline

```
Resume → Tokenize → Remove Stop Words → Stem → TF‑IDF Vector   ─┐
                                                                ├─→ Cosine Similarity → Ranked Results
Job Desc → Tokenize → Remove Stop Words → Stem → TF‑IDF Vector ─┘
```

### What "From Scratch" Means

The NLP and ML algorithms below use **only** `math`, `re`, and `collections` — no scikit‑learn, no spaCy, no NLTK.

| Algorithm | Formula | Module |
|---|---|---|
| Term Frequency | `TF(t,d) = count(t in d) / total_terms(d)` | `src/vectorization/tf.py` |
| Inverse Document Frequency | `IDF(t) = log((1 + N) / (1 + df(t))) + 1` | `src/vectorization/idf.py` |
| TF‑IDF | `TF‑IDF(t,d) = TF × IDF` | `src/vectorization/tfidf.py` |
| Cosine Similarity | `cos(A,B) = (A·B) / (‖A‖ × ‖B‖)` | `src/similarity/cosine.py` |
| Porter Stemmer | 25 suffix‑stripping rules | `src/preprocessing/stemmer.py` |
| Tokenizer | Regex‑based cleaning | `src/preprocessing/tokenizer.py` |
| Stop Words | Hand‑crafted 120+ word list | `src/preprocessing/stopwords.py` |

Utility libraries (file I/O, UI, charts) are used where they don't replace the learning objective — see `requirements.txt` for the full list.

---

## How to Run

### 1. Installation
Clone the repository and install the dependencies:
```bash
pip install -r requirements.txt
```

### 2. Launch the Recruiter Dashboard (Streamlit)
Run the interactive dashboard locally:
```bash
streamlit run app.py
```
- Open **http://localhost:8501** in your browser.
- Check the **"Use sample Job Description for demo"** and **"Include pre-loaded sample resumes"** checkboxes on the main page to try the pre-loaded data.
- Click **"Run Resume‑Job Matching & Ranking"** to run the matching pipeline.

### 3. Open the Math Walkthrough (Jupyter Notebook)
To inspect the underlying mathematics (tokenization, stemming, TF-IDF vectors, and cosine similarity) step-by-step:
```bash
jupyter notebook notebooks/walkthrough.ipynb
```

### 4. Run the Test Suite
To execute all unit and validation tests proving correctness against scikit-learn:
```bash
python -m pytest tests/ -v
```

## Project Structure

```
├── app.py                        # Streamlit dashboard
├── requirements.txt
│
├── src/
│   ├── preprocessing/            # Tokenizer · Stop words · Stemmer
│   ├── vectorization/            # TF · IDF · TF‑IDF
│   ├── similarity/               # Cosine similarity · Ranking pipeline
│   └── parsers/                  # PDF & DOCX text extraction
│
├── data/
│   ├── sample_resumes/           # 5 sample resumes (txt)
│   ├── sample_job_descriptions/  # 3 sample JDs (txt)
│   └── skills_database.json      # 200+ skills across 8 categories
│
├── notebooks/
│   └── walkthrough.ipynb         # Step‑by‑step Jupyter walkthrough
│
└── tests/
    ├── test_preprocessing.py
    ├── test_tfidf.py
    ├── test_cosine.py
    └── test_validate_sklearn.py  # Proves our output matches sklearn
```

---

## Dashboard Features

| Tab | What It Shows |
|---|---|
| **Candidate Rankings** | Leaderboard sorted by cosine similarity, expandable skill details per candidate, CSV export |
| **Skills Analysis** | Matched vs. missing vs. bonus skills for every resume against the JD |
| **How It Works** | Live TF‑IDF vectors, dot‑product breakdown, and per‑term contribution — full math transparency |
| **Visualizations** | Score comparison bar chart, skill gap grouped bars, similarity‑vs‑skill scatter plot |

---

## Validation

`test_validate_sklearn.py` runs the same data through our from‑scratch code **and** scikit‑learn, then asserts the cosine similarity values and ranking order match. This proves correctness without relying on sklearn at runtime.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.8+ |
| UI | Streamlit |
| File Parsing | pdfplumber, python‑docx |
| Charts | Plotly |
| Notebook | Jupyter |
| Testing | pytest |
