"""
Resume Ranker — Main Matching Pipeline
========================================
This module orchestrates the entire resume-job matching pipeline:
    1. Preprocess both the JD and all resumes
    2. Build TF-IDF vectors
    3. Calculate cosine similarity between JD and each resume
    4. Extract matched/missing skills
    5. Return ranked results

This is the "brain" that connects all from-scratch modules together.
"""

import json
import os
import re
from functools import lru_cache

from ..preprocessing.tokenizer import tokenize
from ..preprocessing.stopwords import remove_stopwords
from ..preprocessing.stemmer import stem_tokens
from ..vectorization.tfidf import build_tfidf_matrix, compute_tfidf, get_top_terms
from ..vectorization.idf import compute_idf
from .cosine import cosine_similarity, cosine_similarity_detailed


# Path to skills database
_SKILLS_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'data', 'skills_database.json'
)


def preprocess(text):
    """
    Full preprocessing pipeline: tokenize → remove stop words → stem.

    Args:
        text (str): Raw text

    Returns:
        list[str]: Preprocessed tokens ready for TF-IDF
    """
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = stem_tokens(tokens)
    return tokens


def load_skills_database():
    """
    Load the curated skills database from JSON file.

    Returns:
        dict: Skills organized by category
              {'programming': ['python', 'java', ...], 'ml_ai': [...], ...}
    """
    try:
        with open(_SKILLS_DB_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        # Return a minimal fallback if file not found
        return {
            'programming': ['python', 'java', 'javascript', 'sql', 'c', 'cpp', 'r'],
            'ml_ai': ['tensorflow', 'pytorch', 'scikit', 'keras', 'nlp'],
            'web': ['react', 'angular', 'node', 'django', 'flask', 'html', 'css'],
            'cloud': ['aws', 'azure', 'gcp', 'docker', 'kubernetes'],
            'databases': ['mysql', 'postgresql', 'mongodb', 'redis'],
            'tools': ['git', 'linux', 'jira', 'jenkins', 'ci', 'cd'],
        }


# Skills that are also everyday words only count when written the way the
# language is written ("R", "Go"), so "go to market" or "R&D" don't match.
_CASE_SENSITIVE = {'r': 'R', 'go': 'Go'}


@lru_cache(maxsize=None)
def _skill_pattern(skill):
    """Regex that matches a skill as a whole word, never inside another word.

    Plain substring matching found "java" in "javascript", "go" in "google"
    and "r" in almost every resume. Lookarounds are used instead of \\b so
    skills with symbols ("c++", "c#", ".net", "node.js") still match.
    """
    skill_lower = skill.lower()
    if skill_lower in _CASE_SENSITIVE:
        return re.compile(r'(?<![A-Za-z0-9])' + re.escape(_CASE_SENSITIVE[skill_lower]) + r'(?![A-Za-z0-9+#&])')
    return re.compile(r'(?<![a-z0-9])' + re.escape(skill_lower) + r'(?![a-z0-9+#])', re.IGNORECASE)


def extract_skills(text, skills_db=None):
    """
    Extract skills found in text by matching against the skills database.

    This uses keyword matching (not ML) — which is practical and what
    real ATS systems do for skill extraction.

    Args:
        text (str): Raw text from a resume or JD
        skills_db (dict, optional): Skills database. Loads default if None.

    Returns:
        dict: {
            'all_skills': list[str],            # All skills found
            'by_category': dict[str, list[str]], # Skills grouped by category
        }
    """
    if skills_db is None:
        skills_db = load_skills_database()

    found_skills = {}

    for category, skills_list in skills_db.items():
        category_found = [skill for skill in skills_list if _skill_pattern(skill).search(text)]
        if category_found:
            found_skills[category] = category_found

    all_found = []
    for skills in found_skills.values():
        all_found.extend(skills)

    return {
        'all_skills': sorted(set(all_found)),
        'by_category': found_skills,
    }


def compute_skill_gap(jd_skills, resume_skills):
    """
    Compare JD required skills vs resume skills to find gaps.

    Args:
        jd_skills (dict): Output of extract_skills() for the JD
        resume_skills (dict): Output of extract_skills() for a resume

    Returns:
        dict: {
            'matched': list[str],  # Skills in both JD and resume
            'missing': list[str],  # Skills in JD but not in resume
            'extra': list[str],    # Skills in resume but not in JD
            'match_rate': float,   # Percentage of JD skills matched (0-100)
        }
    """
    jd_set = set(s.lower() for s in jd_skills['all_skills'])
    resume_set = set(s.lower() for s in resume_skills['all_skills'])

    matched = sorted(jd_set & resume_set)
    missing = sorted(jd_set - resume_set)
    extra = sorted(resume_set - jd_set)

    match_rate = (len(matched) / len(jd_set) * 100) if jd_set else 0.0

    return {
        'matched': matched,
        'missing': missing,
        'extra': extra,
        'match_rate': round(match_rate, 1),
    }


def rank_resumes(job_description, resumes, detailed=False):
    """
    Main function: Rank all resumes against a job description.

    Pipeline:
        1. Preprocess JD and all resumes (tokenize → stop words → stem)
        2. Build corpus (JD + all resumes) for IDF calculation
        3. Compute TF-IDF vectors for everything
        4. Calculate cosine similarity between JD and each resume
        5. Extract skills and compute skill gaps
        6. Sort by similarity score (descending)

    Args:
        job_description (str): Raw job description text
        resumes (dict[str, str]): Mapping of {resume_name: raw_text}
        detailed (bool): If True, include detailed cosine similarity breakdown

    Returns:
        list[dict]: Ranked results, each containing:
            - name (str): Resume identifier
            - score (float): Cosine similarity (0 to 1)
            - score_pct (float): Score as percentage (0 to 100)
            - matched_skills (list): Skills found in both JD and resume
            - missing_skills (list): Skills in JD but not in resume
            - extra_skills (list): Skills in resume but not in JD
            - skill_match_rate (float): % of JD skills matched
            - top_terms (list): Top TF-IDF terms from the resume
            - details (dict, optional): Step-by-step cosine calculation

    Example:
        >>> results = rank_resumes(
        ...     "Looking for Python ML engineer with TensorFlow",
        ...     {"alice.pdf": "Python developer with TensorFlow...",
        ...      "bob.pdf": "Java developer with Spring Boot..."}
        ... )
        >>> results[0]['name']
        'alice.pdf'
        >>> results[0]['score_pct']
        78.3
    """
    if not job_description or not resumes:
        return []

    # ---- Step 1: Preprocess ----
    jd_tokens = preprocess(job_description)

    resume_data = {}
    for name, text in resumes.items():
        resume_data[name] = {
            'tokens': preprocess(text),
            'raw_text': text,
        }

    # ---- Step 2: Build corpus for IDF ----
    # The corpus includes the JD + all resumes
    all_documents = [jd_tokens] + [r['tokens'] for r in resume_data.values()]

    # ---- Step 3: Compute IDF across entire corpus ----
    idf_scores = compute_idf(all_documents)

    # ---- Step 4: Compute TF-IDF vectors ----
    jd_vector = compute_tfidf(jd_tokens, idf_scores)

    # ---- Step 5: Extract skills from JD ----
    skills_db = load_skills_database()
    jd_skills = extract_skills(job_description, skills_db)

    # ---- Step 6: Score each resume ----
    results = []
    for name, data in resume_data.items():
        # Compute TF-IDF vector for this resume
        resume_vector = compute_tfidf(data['tokens'], idf_scores)

        # Calculate cosine similarity
        if detailed:
            details = cosine_similarity_detailed(jd_vector, resume_vector)
            score = details['similarity']
        else:
            score = cosine_similarity(jd_vector, resume_vector)
            details = None

        # Extract skills from resume
        resume_skills = extract_skills(data['raw_text'], skills_db)

        # Compute skill gap
        skill_gap = compute_skill_gap(jd_skills, resume_skills)

        # Get top TF-IDF terms
        top_terms = get_top_terms(resume_vector, n=10)

        results.append({
            'name': name,
            'score': round(score, 6),
            'score_pct': round(score * 100, 1),
            'matched_skills': skill_gap['matched'],
            'missing_skills': skill_gap['missing'],
            'extra_skills': skill_gap['extra'],
            'skill_match_rate': skill_gap['match_rate'],
            'top_terms': top_terms,
            'resume_skills': resume_skills,
            'details': details,
        })

    # ---- Step 7: Sort by score (highest first) ----
    results.sort(key=lambda x: x['score'], reverse=True)

    return results
