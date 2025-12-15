import multiprocessing
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple, Union

import bm25s
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer


class BM25Handler:
    """
    Handler class for BM25 embeddings with support for incremental document addition.

    Features:
    - Initial corpus fitting using optimized parallel processing
    - Add new documents without recomputing the entire corpus
    - Reuse existing vocabulary and embeddings
    - Query and retrieve similar documents
    """

    def __init__(
        self,
        max_features: int = 5000,
        method: str = "lucene",
        k1: float = 1.5,
        b: float = 0.75,
        stopwords: Optional[set] = None,
        n_jobs: int = -1,
    ):
        """
        Initialize BM25Handler with configuration parameters.

        Args:
            max_features: Maximum number of features for vocabulary
            method: BM25 variant - "lucene", "robertson", "atire", "bm25+", "bm25l"
            k1: BM25 k1 parameter (term frequency saturation)
            b: BM25 b parameter (length normalization)
            stopwords: Set of stopwords to filter
            n_jobs: Number of parallel jobs (-1 for all CPUs)
        """
        self.max_features = max_features
        self.method = method
        self.k1 = k1
        self.b = b
        self.stopwords = stopwords
        self.n_jobs = n_jobs if n_jobs != -1 else multiprocessing.cpu_count()

        # State variables
        self.bm25_matrix = None
        self.vectorizer = None
        self.retriever = None
        self.vocab = None
        self.corpus_tokens = []
        self.corpus_texts = []
        self.is_fitted = False

    def compute_bm25_embeddings(
        self, df: pd.DataFrame, text_col: str, max_feature: int = None
    ) -> Tuple:
        """
        Compute BM25 embeddings with optimized parallel processing.

        Args:
            df: DataFrame with text data
            text_col: Column name containing text

        Returns:
            bm25_matrix: BM25 sparse matrix (n_docs x n_features)
            vectorizer: Fitted CountVectorizer (for feature names/vocab)
            retriever: bm25s BM25 object (for querying later)
        """
        # Auto-calculate max_features if not provided
        max_features = self.max_features if max_feature is None else max_feature
        if max_features is None:
            avg_tokens = df[text_col].astype(str).str.split().str.len().mean()
            max_features = min(5000, int(avg_tokens * 1.5))
            print(
                f"Auto-selected max_features: {max_features} (based on avg tokens: {int(avg_tokens)})"
            )

        start_time = time.time()

        corpus_texts = df[text_col].astype(str).tolist()
        n_docs = len(corpus_texts)

        print(f"Processing {n_docs:,} documents with {self.n_jobs} CPU cores...")

        # Fast tokenization with parallel processing
        print("Tokenizing corpus in parallel...")
        if self.n_jobs > 1:
            with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                corpus_tokens = list(executor.map(str.split, corpus_texts))
        else:
            corpus_tokens = [doc.split() for doc in corpus_texts]

        tokenize_time = time.time()
        print(f"  Tokenization: {tokenize_time - start_time:.1f}s")

        # Filter stopwords
        if self.stopwords:
            if self.n_jobs > 1:

                def filter_stopwords(tokens):
                    return [t for t in tokens if t not in self.stopwords]

                with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                    corpus_tokens = list(executor.map(filter_stopwords, corpus_tokens))
            else:
                corpus_tokens = [
                    [t for t in doc if t not in self.stopwords] for doc in corpus_tokens
                ]
            print(f"Filtered {len(self.stopwords)} stopwords")

        # Build vocabulary (vectorized for speed)
        print("Building vocabulary...")

        # Simple single-threaded vocabulary building (fast enough)
        doc_freq = defaultdict(int)
        for doc_tokens in corpus_tokens:
            for token in set(doc_tokens):
                doc_freq[token] += 1

        # Filter by document frequency
        min_df = 2
        max_df = 0.95
        max_doc_freq = int(n_docs * max_df)

        valid_tokens = {
            token for token, freq in doc_freq.items() if freq >= min_df and freq <= max_doc_freq
        }

        # Limit to top max_features
        if max_features and len(valid_tokens) > max_features:
            token_freq = Counter()
            for doc_tokens in corpus_tokens:
                token_freq.update([t for t in doc_tokens if t in valid_tokens])

            top_tokens = set([token for token, _ in token_freq.most_common(max_features)])
            valid_tokens = top_tokens

        # Filter corpus in parallel
        def filter_valid_tokens(tokens):
            return [t for t in tokens if t in valid_tokens]

        if self.n_jobs > 1:
            with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                corpus_tokens = list(executor.map(filter_valid_tokens, corpus_tokens))
        else:
            corpus_tokens = [filter_valid_tokens(doc) for doc in corpus_tokens]

        # Build vocabulary
        vocab = {token: idx for idx, token in enumerate(sorted(valid_tokens))}
        n_features = len(vocab)

        vocab_time = time.time()
        print(f"  Vocabulary ({n_features:,} features): {vocab_time - tokenize_time:.1f}s")

        # Convert to IDs in parallel
        print("Converting tokens to IDs...")

        def tokens_to_ids(tokens):
            return [vocab[t] for t in tokens if t in vocab]

        if self.n_jobs > 1:
            with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                corpus_token_ids = list(executor.map(tokens_to_ids, corpus_tokens))
        else:
            corpus_token_ids = [tokens_to_ids(doc) for doc in corpus_tokens]

        # Create and index BM25
        print("Indexing with BM25...")
        retriever = bm25s.BM25(method=self.method, k1=self.k1, b=self.b)
        retriever.index(corpus_token_ids)

        index_time = time.time()
        print(f"  BM25 indexing: {index_time - vocab_time:.1f}s")

        # Build sparse matrix with THREADED processing (Streamlit-safe)
        print("Building sparse matrix in parallel...")

        def process_batch(batch_indices):
            """Process a batch of documents - now pickleable!"""
            rows_local, cols_local, data_local = [], [], []

            for doc_idx in batch_indices:
                if corpus_token_ids[doc_idx]:
                    doc_scores = retriever.get_scores(corpus_token_ids[doc_idx])

                    # Vectorized non-zero extraction
                    nz_mask = doc_scores > 0
                    if nz_mask.any():
                        nz_indices = np.where(nz_mask)[0]
                        # Clip to valid range
                        valid_mask = nz_indices < n_features
                        nz_indices = nz_indices[valid_mask]

                        rows_local.extend([doc_idx] * len(nz_indices))
                        cols_local.extend(nz_indices.tolist())
                        data_local.extend(doc_scores[nz_indices].tolist())

            return rows_local, cols_local, data_local

        # Split into batches for parallel processing
        if self.n_jobs > 1:
            batch_size = max(n_docs // (self.n_jobs * 4), 100)
            batches = [
                list(range(i, min(i + batch_size, n_docs))) for i in range(0, n_docs, batch_size)
            ]

            # Use ThreadPoolExecutor (works with Streamlit!)
            with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                results = list(executor.map(process_batch, batches))

            # Merge results
            rows, cols, data = [], [], []
            for r, c, d in results:
                rows.extend(r)
                cols.extend(c)
                data.extend(d)
        else:
            rows, cols, data = process_batch(list(range(n_docs)))

        matrix_time = time.time()
        print(f"  Sparse matrix construction: {matrix_time - index_time:.1f}s")

        # Create sparse matrix
        bm25_matrix = sp.csr_matrix((data, (rows, cols)), shape=(n_docs, n_features))

        # Create compatible vectorizer
        vectorizer = CountVectorizer()
        vectorizer.vocabulary_ = vocab
        vectorizer.max_features = n_features

        # Statistics
        total_time = time.time() - start_time
        sparsity = 1.0 - (bm25_matrix.nnz / (bm25_matrix.shape[0] * bm25_matrix.shape[1]))
        mem_mb = (
            (bm25_matrix.data.nbytes + bm25_matrix.indices.nbytes + bm25_matrix.indptr.nbytes)
            / 1024
            / 1024
        )

        print(f"\n{'='*60}")
        print(f"BM25 Computation Complete!")
        print(f"{'='*60}")
        print(f"Total time: {total_time:.1f}s")
        print(f"Matrix shape: {bm25_matrix.shape}")
        print(f"Non-zero elements: {bm25_matrix.nnz:,}")
        print(f"Sparsity: {sparsity*100:.2f}%")
        print(f"Memory: {mem_mb:.1f} MB")
        print(f"CPUs used: {self.n_jobs}")
        print(f"Speed: {n_docs/total_time:.0f} docs/sec")
        print(f"{'='*60}")

        # Store corpus tokens for later use
        self.corpus_tokens = corpus_tokens

        return bm25_matrix, vectorizer, retriever

    def fit(self, df: pd.DataFrame, text_col: str) -> "BM25Handler":
        """
        Fit BM25 on initial corpus using the optimized compute_bm25_embeddings method.

        Args:
            df: DataFrame with text data
            text_col: Column name containing text

        Returns:
            self for method chaining
        """
        print(f"Fitting BM25Handler on {len(df)} documents...")

        # Use the optimized method
        self.bm25_matrix, self.vectorizer, self.retriever = self.compute_bm25_embeddings(
            df=df, text_col=text_col
        )

        # Store corpus
        self.corpus_texts = df[text_col].astype(str).tolist()
        self.vocab = self.vectorizer.vocabulary_

        self.is_fitted = True
        print("✓ BM25Handler fitted successfully!\n")
        return self

    def add_documents(
        self, new_texts: Union[List[str], pd.Series], reindex: bool = True
    ) -> sp.csr_matrix:
        """
        Add new documents and compute their BM25 embeddings using existing vocabulary.

        Args:
            new_texts: List or Series of new text documents
            reindex: Whether to rebuild the BM25 retriever with all documents

        Returns:
            Sparse matrix of new document embeddings (n_new_docs x n_features)
        """
        if not self.is_fitted:
            raise ValueError(
                "BM25Handler must be fitted before adding documents. Call fit() first."
            )

        if isinstance(new_texts, pd.Series):
            new_texts = new_texts.tolist()

        n_new = len(new_texts)
        print(f"Adding {n_new} new documents to BM25Handler...")

        # Tokenize new documents
        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            new_tokens = list(executor.map(str.split, new_texts))

        # Filter stopwords
        if self.stopwords:
            with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                new_tokens = list(
                    executor.map(
                        lambda tokens: [t for t in tokens if t not in self.stopwords], new_tokens
                    )
                )

        # Convert to IDs using EXISTING vocabulary (OOV tokens ignored)
        new_token_ids = self._tokens_to_ids(new_tokens)

        # Compute BM25 scores for new documents
        n_features = len(self.vocab)

        rows, cols, data = [], [], []
        for doc_idx, token_ids in enumerate(new_token_ids):
            if token_ids:
                doc_scores = self.retriever.get_scores(token_ids)

                # Extract non-zero scores
                nz_mask = doc_scores > 0
                if nz_mask.any():
                    nz_indices = np.where(nz_mask)[0]
                    valid_mask = nz_indices < n_features
                    nz_indices = nz_indices[valid_mask]

                    rows.extend([doc_idx] * len(nz_indices))
                    cols.extend(nz_indices.tolist())
                    data.extend(doc_scores[nz_indices].tolist())

        # Create sparse matrix for new documents
        new_matrix = sp.csr_matrix((data, (rows, cols)), shape=(n_new, n_features))

        # Append to existing matrix
        self.bm25_matrix = sp.vstack([self.bm25_matrix, new_matrix], format="csr")

        # Update corpus
        self.corpus_texts.extend(new_texts)
        self.corpus_tokens.extend(new_tokens)

        # Optionally reindex retriever with all documents
        if reindex:
            print("Reindexing BM25 retriever with all documents...")
            all_token_ids = self._tokens_to_ids(self.corpus_tokens)
            self.retriever = bm25s.BM25(method=self.method, k1=self.k1, b=self.b)
            self.retriever.index(all_token_ids)

        sparsity = 1 - self.bm25_matrix.nnz / (
            self.bm25_matrix.shape[0] * self.bm25_matrix.shape[1]
        )
        print(f"✓ Added {n_new} documents. Total corpus: {len(self.corpus_texts)} documents")
        print(f"  New matrix shape: {self.bm25_matrix.shape}")
        print(f"  Sparsity: {sparsity * 100:.2f}%\n")

        return new_matrix

    def query(
        self, query_text: str, top_k: int = 10, return_scores: bool = True
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Query the BM25 index and retrieve top-k similar documents.

        Args:
            query_text: Query string
            top_k: Number of top results to return
            return_scores: Whether to return scores along with indices

        Returns:
            If return_scores=True: (indices, scores) tuple
            If return_scores=False: indices only
        """
        if not self.is_fitted:
            raise ValueError("BM25Handler must be fitted before querying.")

        # Tokenize query
        query_tokens = query_text.split()

        # Filter stopwords
        if self.stopwords:
            query_tokens = [t for t in query_tokens if t not in self.stopwords]

        # Convert to IDs
        query_token_ids = [self.vocab[t] for t in query_tokens if t in self.vocab]

        if not query_token_ids:
            print("Warning: No valid tokens in query after filtering.")
            return (np.array([]), np.array([])) if return_scores else np.array([])

        # Get scores for all documents
        scores = self.retriever.get_scores(query_token_ids)

        # Get top-k indices
        top_k = min(top_k, len(scores))
        top_indices = np.argsort(scores)[-top_k:][::-1]
        top_scores = scores[top_indices]

        if return_scores:
            return top_indices, top_scores
        return top_indices

    def get_embeddings(self, indices: Optional[List[int]] = None) -> sp.csr_matrix:
        """
        Get BM25 embeddings for specific documents or all documents.

        Args:
            indices: List of document indices (None for all documents)

        Returns:
            Sparse matrix of embeddings
        """
        if not self.is_fitted:
            raise ValueError("BM25Handler must be fitted before getting embeddings.")

        if indices is None:
            return self.bm25_matrix
        return self.bm25_matrix[indices]

    def get_document(self, index: int) -> str:
        """Get original text of a document by index."""
        if not self.is_fitted:
            raise ValueError("BM25Handler must be fitted.")
        return self.corpus_texts[index]

    def _tokens_to_ids(self, token_lists: List[List[str]]) -> List[List[int]]:
        """Convert token lists to ID lists using existing vocabulary."""

        def tokens_to_ids(tokens):
            return [self.vocab[t] for t in tokens if t in self.vocab]

        if self.n_jobs > 1:
            with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                return list(executor.map(tokens_to_ids, token_lists))
        return [tokens_to_ids(tokens) for tokens in token_lists]

    def save_state(self) -> dict:
        """Save handler state for persistence."""
        return {
            "bm25_matrix": self.bm25_matrix,
            "vocab": self.vocab,
            "corpus_texts": self.corpus_texts,
            "corpus_tokens": self.corpus_tokens,
            "max_features": self.max_features,
            "method": self.method,
            "k1": self.k1,
            "b": self.b,
            "is_fitted": self.is_fitted,
        }

    def load_state(self, state: dict):
        """Load handler state from saved state."""
        self.bm25_matrix = state["bm25_matrix"]
        self.vocab = state["vocab"]
        self.corpus_texts = state["corpus_texts"]
        self.corpus_tokens = state["corpus_tokens"]
        self.max_features = state["max_features"]
        self.method = state["method"]
        self.k1 = state["k1"]
        self.b = state["b"]
        self.is_fitted = state["is_fitted"]

        # Recreate vectorizer
        self.vectorizer = CountVectorizer()
        self.vectorizer.vocabulary_ = self.vocab
        self.vectorizer.max_features = len(self.vocab)

        # Recreate retriever
        if self.is_fitted:
            corpus_token_ids = self._tokens_to_ids(self.corpus_tokens)
            self.retriever = bm25s.BM25(method=self.method, k1=self.k1, b=self.b)
            self.retriever.index(corpus_token_ids)

    def __repr__(self):
        if not self.is_fitted:
            return f"BM25Handler(not fitted)"
        return (
            f"BM25Handler(docs={len(self.corpus_texts)}, "
            f"features={len(self.vocab)}, method='{self.method}')"
        )
