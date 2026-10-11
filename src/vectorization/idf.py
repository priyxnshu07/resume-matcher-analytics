"""
Inverse Document Frequency (IDF) — FROM SCRATCH
=================================================
IDF measures how RARE or UNIQUE a word is across all documents.
It's the second half of the TF-IDF calculation.

The key insight: Words that appear in many documents are less useful
for distinguishing between them. IDF penalizes common words and
rewards rare, distinctive ones.

Formula (smoothed, the same as scikit-learn's default):
    IDF(t) = log((1 + N) / (1 + df(t))) + 1

    Where:
        N     = Total number of documents in the corpus
        df(t) = Number of documents that contain term t
        +1s   = Smoothing: no division by zero, and IDF is always > 0
        log   = Natural logarithm (compresses the scale)

Example (3 documents):
    Doc1: "python machine learning"
    Doc2: "python data science"
    Doc3: "java machine learning"

    N = 3 documents

    "python"   appears in Doc1, Doc2     → df=2 → IDF = log(4/3) + 1 = 1.288
    "machine"  appears in Doc1, Doc3     → df=2 → IDF = log(4/3) + 1 = 1.288
    "java"     appears in Doc3 only      → df=1 → IDF = log(4/2) + 1 = 1.693
    "data"     appears in Doc2 only      → df=1 → IDF = log(4/2) + 1 = 1.693
    "science"  appears in Doc2 only      → df=1 → IDF = log(4/2) + 1 = 1.693

    → "java", "data", "science" are MORE important (rare = distinctive!)
    → "python", "machine" are LESS important (common across docs)

Implementation: Uses only math.log. No sklearn!
"""

import math


def compute_idf(documents):
    """
    Calculate Inverse Document Frequency for every word across all documents.

    Formula (smoothed): IDF(t) = log((1 + N) / (1 + df(t))) + 1

    Words that appear in fewer documents get higher IDF scores,
    meaning they are more useful for distinguishing documents.

    Args:
        documents (list[list[str]]): List of documents, where each document
                                      is a list of preprocessed tokens.
                                      Example: [['python', 'ml'], ['java', 'ml']]

    Returns:
        dict[str, float]: Mapping of each word to its IDF score

    Example:
        >>> docs = [['python', 'ml'], ['java', 'ml'], ['python', 'web']]
        >>> idf = compute_idf(docs)
        >>> round(idf['ml'], 3)     # appears in 2/3 docs → lower IDF
        1.288
        >>> round(idf['java'], 3)   # appears in 1/3 docs → higher IDF
        1.693

    Why smoothed: the plain log(N / (1 + df)) is 0 or negative for common
    words. With a JD and one resume (N = 2), every word unique to one document
    gets IDF 0, so only shared words count and any overlap scores 100%.
    The smoothed form is always > 0 and is the same formula scikit-learn uses.
    """
    if not documents:
        return {}

    N = len(documents)  # Total number of documents

    # Step 1: Count Document Frequency (df) — how many documents contain each word
    # IMPORTANT: Count each word only ONCE per document (use set)
    doc_freq = {}
    for doc_tokens in documents:
        unique_words = set(doc_tokens)  # Deduplicate within document
        for word in unique_words:
            doc_freq[word] = doc_freq.get(word, 0) + 1

    # Step 2: Calculate IDF for each word
    idf = {}
    for word, df in doc_freq.items():
        # Formula: IDF(t) = log((1 + N) / (1 + df(t))) + 1
        # Adding 1 to N and df acts as if one extra document contained every
        # word (no division by zero); the final +1 keeps every IDF positive.
        idf[word] = math.log((1 + N) / (1 + df)) + 1

    return idf


def compute_document_frequency(documents):
    """
    Calculate raw Document Frequency — just the count of documents containing each word.

    Useful for debugging and for the Jupyter notebook walkthrough.

    Args:
        documents (list[list[str]]): List of token lists

    Returns:
        dict[str, int]: Mapping of each word to number of documents it appears in

    Example:
        >>> docs = [['python', 'ml'], ['java', 'ml']]
        >>> compute_document_frequency(docs)
        {'python': 1, 'ml': 2, 'java': 1}
    """
    doc_freq = {}
    for doc_tokens in documents:
        unique_words = set(doc_tokens)
        for word in unique_words:
            doc_freq[word] = doc_freq.get(word, 0) + 1
    return doc_freq
