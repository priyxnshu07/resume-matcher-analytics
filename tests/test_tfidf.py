"""Tests for TF, IDF, and TF-IDF calculations."""

import sys
import os
import math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.vectorization.tf import compute_tf, compute_raw_tf
from src.vectorization.idf import compute_idf, compute_document_frequency
from src.vectorization.tfidf import compute_tfidf, build_tfidf_matrix, get_top_terms


class TestTermFrequency:
    """Tests for from-scratch TF calculation."""

    def test_basic_tf(self):
        tokens = ['python', 'machine', 'learning', 'python', 'data', 'python']
        tf = compute_tf(tokens)
        assert abs(tf['python'] - 3/6) < 1e-10
        assert abs(tf['machine'] - 1/6) < 1e-10
        assert abs(tf['data'] - 1/6) < 1e-10

    def test_single_word(self):
        tf = compute_tf(['python'])
        assert tf['python'] == 1.0

    def test_empty_tokens(self):
        assert compute_tf([]) == {}

    def test_all_same_word(self):
        tf = compute_tf(['python', 'python', 'python'])
        assert tf['python'] == 1.0

    def test_tf_sums_to_one(self):
        tokens = ['python', 'java', 'sql', 'python', 'react']
        tf = compute_tf(tokens)
        assert abs(sum(tf.values()) - 1.0) < 1e-10

    def test_raw_tf(self):
        tokens = ['python', 'java', 'python']
        raw = compute_raw_tf(tokens)
        assert raw['python'] == 2
        assert raw['java'] == 1


class TestInverseDocumentFrequency:
    """Tests for from-scratch IDF calculation."""

    def test_basic_idf(self):
        docs = [
            ['python', 'ml'],
            ['java', 'ml'],
            ['python', 'web'],
        ]
        idf = compute_idf(docs)

        # IDF = log((1 + N) / (1 + df)) + 1
        # 'ml' appears in 2/3 docs → log(4/3) + 1
        assert abs(idf['ml'] - (math.log(4/3) + 1)) < 1e-10

        # 'java' appears in 1/3 docs → log(4/2) + 1
        assert abs(idf['java'] - (math.log(4/2) + 1)) < 1e-10

    def test_rare_word_higher_idf(self):
        docs = [
            ['python', 'ml', 'data'],
            ['python', 'ml', 'web'],
            ['python', 'ml', 'cloud'],
        ]
        idf = compute_idf(docs)

        # 'python' and 'ml' appear in all docs → low IDF
        # 'data', 'web', 'cloud' appear in 1 doc each → high IDF
        assert idf['data'] > idf['python']
        assert idf['cloud'] > idf['ml']

    def test_word_in_all_docs_low_idf(self):
        docs = [['common'], ['common'], ['common']]
        idf = compute_idf(docs)
        # log(4/4) + 1 = 1: the lowest possible IDF, but never zero or negative,
        # so a word shared by every document still counts a little
        assert idf['common'] == 1.0

    def test_empty_corpus(self):
        assert compute_idf([]) == {}

    def test_document_frequency(self):
        docs = [['python', 'ml'], ['java', 'ml']]
        df = compute_document_frequency(docs)
        assert df['ml'] == 2
        assert df['python'] == 1
        assert df['java'] == 1

    def test_duplicate_words_in_doc_counted_once(self):
        docs = [['python', 'python', 'python'], ['java']]
        df = compute_document_frequency(docs)
        assert df['python'] == 1  # Only counted once per document!


class TestTFIDF:
    """Tests for the combined TF-IDF calculation."""

    def test_basic_tfidf(self):
        docs = [['python', 'ml'], ['java', 'ml']]
        idf = compute_idf(docs)
        tfidf = compute_tfidf(['python', 'ml'], idf)

        # 'python' has TF=0.5 and IDF=log(3/2)+1
        assert abs(tfidf['python'] - 0.5 * (math.log(3/2) + 1)) < 1e-10

        # 'ml' is in both docs: lower weight than 'python', but still positive
        assert 0 < tfidf['ml'] < tfidf['python']

    def test_build_matrix(self):
        docs = [['python', 'ml'], ['java', 'ml'], ['python', 'web']]
        vectors, idf = build_tfidf_matrix(docs)
        assert len(vectors) == 3
        assert isinstance(idf, dict)

    def test_get_top_terms(self):
        vec = {'python': 0.3, 'ml': 0.5, 'data': 0.1, 'java': 0.0}
        top = get_top_terms(vec, n=2)
        assert len(top) == 2
        assert top[0][0] == 'ml'  # Highest score first
        assert top[1][0] == 'python'

    def test_empty_document(self):
        assert compute_tfidf([], {}) == {}


# Run tests
if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v'])
