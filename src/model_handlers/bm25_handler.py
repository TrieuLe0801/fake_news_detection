import multiprocessing
import pickle
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional, Tuple, Union

import bm25s
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer


class BM25Handler:
    """
    Handler class for BM25 embeddings with parallel processing and persistence.

    Features:
    - Optimized parallel processing for fitting
    - Save/load to disk for reuse
    - Add new documents incrementally
    - Remove documents by indices
    - Query and retrieve similar documents
    - Transform new texts to BM25 vectors
    """

    def __init__(
        self,
        max_features: int = 5000,
        method: str = "lucene",
        k1: float = 1.5,
        b: float = 0.75,
        stopwords: Optional[set] = None,
        min_df: int = 2,
        max_df: float = 0.95,
        n_jobs: int = -1,
    ):
        self.max_features = max_features
        self.method = method
        self.k1 = k1
        self.b = b
        self.min_df = min_df
        self.max_df = max_df
        self.stopwords = stopwords or set()
        self.n_jobs = n_jobs if n_jobs != -1 else multiprocessing.cpu_count()

        # State variables
        self.bm25_matrix: Optional[sp.csr_matrix] = None
        self.vectorizer: Optional[CountVectorizer] = None
        self.retriever: Optional[bm25s.BM25] = None
        self.vocab: Optional[dict] = None
        self.corpus_tokens: List[List[str]] = []
        self.corpus_texts: List[str] = []
        self.is_fitted: bool = False
        self.n_docs: int = 0

    # ==================== TOKENIZATION ====================

    def _tokenize(self, texts: List[str]) -> List[List[str]]:
        """Tokenize texts in parallel and apply stopwords."""
        if self.n_jobs > 1:
            with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                tokens = list(executor.map(str.split, texts))
        else:
            tokens = [text.split() for text in texts]

        if self.stopwords:

            def filter_sw(doc):
                return [t for t in doc if t not in self.stopwords]

            if self.n_jobs > 1:
                with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                    tokens = list(executor.map(filter_sw, tokens))
            else:
                tokens = [filter_sw(doc) for doc in tokens]

        return tokens

    def _tokens_to_ids(self, token_lists: List[List[str]]) -> List[List[int]]:
        """Convert token lists to ID lists using existing vocabulary."""

        def convert(tokens):
            return [self.vocab[t] for t in tokens if t in self.vocab]

        if self.n_jobs > 1:
            with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                return list(executor.map(convert, token_lists))
        return [convert(tokens) for tokens in token_lists]

    # ==================== VOCABULARY ====================

    def _build_vocabulary(self, corpus_tokens: List[List[str]]) -> dict:
        """Build vocabulary with document frequency filtering."""
        n_docs = len(corpus_tokens)

        # Count document frequency
        doc_freq = defaultdict(int)
        for doc_tokens in corpus_tokens:
            for token in set(doc_tokens):
                doc_freq[token] += 1

        # Filter by min/max df
        max_doc_freq = int(n_docs * self.max_df)
        valid_tokens = {
            token
            for token, freq in doc_freq.items()
            if freq >= self.min_df and freq <= max_doc_freq
        }

        # Limit to max_features
        if self.max_features and len(valid_tokens) > self.max_features:
            token_freq = Counter()
            for doc_tokens in corpus_tokens:
                token_freq.update([t for t in doc_tokens if t in valid_tokens])
            valid_tokens = set([t for t, _ in token_freq.most_common(self.max_features)])

        return {token: idx for idx, token in enumerate(sorted(valid_tokens))}

    # ==================== SPARSE MATRIX ====================

    def _build_sparse_matrix(
        self, corpus_token_ids: List[List[int]], n_docs: int
    ) -> sp.csr_matrix:
        """Build sparse BM25 matrix with parallel processing."""
        n_features = len(self.vocab)
        retriever = self.retriever

        def process_batch(batch_indices):
            rows_local, cols_local, data_local = [], [], []
            for doc_idx in batch_indices:
                if corpus_token_ids[doc_idx]:
                    doc_scores = retriever.get_scores(corpus_token_ids[doc_idx])
                    nz_mask = doc_scores > 0
                    if nz_mask.any():
                        nz_indices = np.where(nz_mask)[0]
                        valid_mask = nz_indices < n_features
                        nz_indices = nz_indices[valid_mask]
                        rows_local.extend([doc_idx] * len(nz_indices))
                        cols_local.extend(nz_indices.tolist())
                        data_local.extend(doc_scores[nz_indices].tolist())
            return rows_local, cols_local, data_local

        if self.n_jobs > 1 and n_docs > 100:
            batch_size = max(n_docs // (self.n_jobs * 4), 100)
            batches = [
                list(range(i, min(i + batch_size, n_docs))) for i in range(0, n_docs, batch_size)
            ]
            with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                results = list(executor.map(process_batch, batches))
            rows, cols, data = [], [], []
            for r, c, d in results:
                rows.extend(r)
                cols.extend(c)
                data.extend(d)
        else:
            rows, cols, data = process_batch(list(range(n_docs)))

        return sp.csr_matrix((data, (rows, cols)), shape=(n_docs, n_features))

    def _print_stats(self):
        """Print matrix statistics."""
        if self.bm25_matrix is None:
            return
        sparsity = 1.0 - (
            self.bm25_matrix.nnz / (self.bm25_matrix.shape[0] * self.bm25_matrix.shape[1])
        )
        mem_mb = (
            (
                self.bm25_matrix.data.nbytes
                + self.bm25_matrix.indices.nbytes
                + self.bm25_matrix.indptr.nbytes
            )
            / 1024
            / 1024
        )

        print(f"\n{'='*60}")
        print(f"BM25 Matrix Statistics:")
        print(f"  Shape: {self.bm25_matrix.shape}")
        print(f"  Non-zero elements: {self.bm25_matrix.nnz:,}")
        print(f"  Sparsity: {sparsity*100:.2f}%")
        print(f"  Memory: {mem_mb:.1f} MB")
        print(f"{'='*60}\n")

    # ==================== FIT ====================

    def fit(self, df: pd.DataFrame, text_col: str) -> "BM25Handler":
        """Fit BM25 on initial corpus with optimized parallel processing."""
        start_time = time.time()

        self.corpus_texts = df[text_col].astype(str).tolist()
        self.n_docs = len(self.corpus_texts)

        print(f"Processing {self.n_docs:,} documents with {self.n_jobs} CPU cores...")

        # Tokenize
        print("Tokenizing corpus...")
        t0 = time.time()
        self.corpus_tokens = self._tokenize(self.corpus_texts)
        print(f"  Tokenization: {time.time() - t0:.1f}s")

        # Build vocabulary
        print("Building vocabulary...")
        t0 = time.time()
        self.vocab = self._build_vocabulary(self.corpus_tokens)
        n_features = len(self.vocab)
        print(f"  Vocabulary ({n_features:,} features): {time.time() - t0:.1f}s")

        # Filter tokens to valid vocabulary
        def filter_valid(tokens):
            return [t for t in tokens if t in self.vocab]

        if self.n_jobs > 1:
            with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                self.corpus_tokens = list(executor.map(filter_valid, self.corpus_tokens))
        else:
            self.corpus_tokens = [filter_valid(doc) for doc in self.corpus_tokens]

        # Convert to IDs
        print("Converting tokens to IDs...")
        t0 = time.time()
        corpus_token_ids = self._tokens_to_ids(self.corpus_tokens)
        print(f"  Token to ID: {time.time() - t0:.1f}s")

        # Create BM25 retriever
        print("Indexing with BM25...")
        t0 = time.time()
        self.retriever = bm25s.BM25(method=self.method, k1=self.k1, b=self.b)
        self.retriever.index(corpus_token_ids)
        print(f"  BM25 indexing: {time.time() - t0:.1f}s")

        # Build sparse matrix
        print("Building sparse matrix...")
        t0 = time.time()
        self.bm25_matrix = self._build_sparse_matrix(corpus_token_ids, self.n_docs)
        print(f"  Sparse matrix: {time.time() - t0:.1f}s")

        # Create vectorizer
        self.vectorizer = CountVectorizer()
        self.vectorizer.vocabulary_ = self.vocab
        self.vectorizer.max_features = n_features

        self.is_fitted = True

        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"BM25 Fitting Complete!")
        print(f"  Total time: {total_time:.1f}s")
        print(f"  Speed: {self.n_docs/total_time:.0f} docs/sec")
        print(f"  CPUs used: {self.n_jobs}")
        self._print_stats()

        return self

    # ==================== TRANSFORM ====================

    def transform(self, texts: Union[str, List[str]]) -> sp.csr_matrix:
        """Transform new text(s) to BM25 vector(s) using existing vocabulary."""
        if not self.is_fitted:
            raise ValueError("BM25Handler must be fitted before transform.")

        if isinstance(texts, str):
            texts = [texts]

        n_features = len(self.vocab)
        tokens_list = self._tokenize(texts)
        token_ids_list = self._tokens_to_ids(tokens_list)

        rows, cols, data = [], [], []
        for doc_idx, token_ids in enumerate(token_ids_list):
            if token_ids:
                doc_scores = self.retriever.get_scores(token_ids)
                nz_mask = doc_scores > 0
                if nz_mask.any():
                    nz_indices = np.where(nz_mask)[0]
                    valid_mask = nz_indices < n_features
                    nz_indices = nz_indices[valid_mask]
                    rows.extend([doc_idx] * len(nz_indices))
                    cols.extend(nz_indices.tolist())
                    data.extend(doc_scores[nz_indices].tolist())

        return sp.csr_matrix((data, (rows, cols)), shape=(len(texts), n_features))

    # ==================== ADD DOCUMENTS ====================

    def add_documents(
        self, new_texts: Union[List[str], pd.Series], rebuild_vocab: bool = False
    ) -> sp.csr_matrix:
        """
        Add new documents to the existing index.

        Args:
            new_texts: List or Series of new text documents
            rebuild_vocab: If True, rebuild vocabulary from all docs (slower)
                          If False, use existing vocab (faster, OOV ignored)

        Returns:
            Sparse matrix of new document embeddings
        """
        if not self.is_fitted:
            raise ValueError("BM25Handler must be fitted before adding documents.")

        if isinstance(new_texts, pd.Series):
            new_texts = new_texts.tolist()

        n_new = len(new_texts)
        print(f"Adding {n_new} new documents...")

        # Tokenize new documents
        new_tokens = self._tokenize(new_texts)

        if rebuild_vocab:
            print("Rebuilding vocabulary from all documents...")
            all_tokens = self.corpus_tokens + new_tokens
            self.vocab = self._build_vocabulary(all_tokens)
            print(f"  New vocabulary size: {len(self.vocab):,}")

            # Update corpus
            self.corpus_texts.extend(new_texts)
            self.corpus_tokens = all_tokens
            self.n_docs = len(self.corpus_tokens)

            # Reindex everything
            all_token_ids = self._tokens_to_ids(self.corpus_tokens)

            print("Rebuilding BM25 index...")
            self.retriever = bm25s.BM25(method=self.method, k1=self.k1, b=self.b)
            self.retriever.index(all_token_ids)

            print("Rebuilding sparse matrix...")
            self.bm25_matrix = self._build_sparse_matrix(all_token_ids, self.n_docs)

            # Update vectorizer
            self.vectorizer.vocabulary_ = self.vocab
            self.vectorizer.max_features = len(self.vocab)

            new_matrix = self.bm25_matrix[-n_new:]
        else:
            print("Using existing vocabulary (set rebuild_vocab=True to update)")

            # Update corpus
            self.corpus_texts.extend(new_texts)
            self.corpus_tokens.extend(new_tokens)
            self.n_docs += n_new

            # Reindex retriever with all documents
            print("Updating BM25 index...")
            all_token_ids = self._tokens_to_ids(self.corpus_tokens)
            self.retriever = bm25s.BM25(method=self.method, k1=self.k1, b=self.b)
            self.retriever.index(all_token_ids)

            # Recompute ALL vectors (IDF changed)
            print("Recomputing all document vectors (IDF updated)...")
            self.bm25_matrix = self._build_sparse_matrix(all_token_ids, self.n_docs)

            new_matrix = self.bm25_matrix[-n_new:]

        print(f"✓ Added {n_new} documents. Total: {self.n_docs}")
        self._print_stats()

        return new_matrix

    # ==================== REMOVE DOCUMENTS ====================

    def remove_documents(self, indices: List[int]) -> "BM25Handler":
        """Remove documents by their indices and rebuild index."""
        if not self.is_fitted:
            raise ValueError("BM25Handler must be fitted before removing documents.")

        indices_set = set(indices)
        n_remove = len(indices_set)

        # Filter corpus
        self.corpus_texts = [t for i, t in enumerate(self.corpus_texts) if i not in indices_set]
        self.corpus_tokens = [t for i, t in enumerate(self.corpus_tokens) if i not in indices_set]
        self.n_docs = len(self.corpus_texts)

        print(f"Removing {n_remove} documents. Rebuilding index...")

        # Reindex
        all_token_ids = self._tokens_to_ids(self.corpus_tokens)
        self.retriever = bm25s.BM25(method=self.method, k1=self.k1, b=self.b)
        self.retriever.index(all_token_ids)

        # Rebuild matrix
        self.bm25_matrix = self._build_sparse_matrix(all_token_ids, self.n_docs)

        print(f"✓ Removed {n_remove} documents. Total: {self.n_docs}")
        self._print_stats()

        return self

    # ==================== QUERY ====================

    def query(
        self, query_text: str, top_k: int = 10, return_scores: bool = True
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Query the BM25 index and retrieve top-k similar documents."""
        if not self.is_fitted:
            raise ValueError("BM25Handler must be fitted before querying.")

        # Tokenize and convert query
        query_tokens = query_text.split()
        if self.stopwords:
            query_tokens = [t for t in query_tokens if t not in self.stopwords]
        query_token_ids = [self.vocab[t] for t in query_tokens if t in self.vocab]

        if not query_token_ids:
            print("Warning: No valid tokens in query after filtering.")
            return (np.array([]), np.array([])) if return_scores else np.array([])

        # Get scores
        scores = self.retriever.get_scores(query_token_ids)

        # Get top-k
        top_k = min(top_k, len(scores))
        top_indices = np.argsort(scores)[-top_k:][::-1]
        top_scores = scores[top_indices]

        return (top_indices, top_scores) if return_scores else top_indices

    # ==================== SAVE / LOAD ====================

    def save(self, directory: str) -> None:
        """Save all components to a directory."""
        if not self.is_fitted:
            raise ValueError("BM25Handler must be fitted before saving.")

        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

        # Save sparse matrix
        sp.save_npz(path / "bm25_matrix.npz", self.bm25_matrix)

        # Save retriever using bm25s built-in method
        self.retriever.save(str(path / "retriever"), corpus=None)

        # Save other components
        components = {
            "vocab": self.vocab,
            "corpus_tokens": self.corpus_tokens,
            "corpus_texts": self.corpus_texts,
            "n_docs": self.n_docs,
            "is_fitted": self.is_fitted,
            "config": {
                "max_features": self.max_features,
                "method": self.method,
                "k1": self.k1,
                "b": self.b,
                "min_df": self.min_df,
                "max_df": self.max_df,
                "stopwords": self.stopwords,
                "n_jobs": self.n_jobs,
            },
        }
        with open(path / "components.pkl", "wb") as f:
            pickle.dump(components, f)

        print(f"✓ Saved BM25Handler to: {directory}")

    @classmethod
    def load(cls, directory: str) -> "BM25Handler":
        """Load all components from a directory."""
        path = Path(directory)

        # Load components
        with open(path / "components.pkl", "rb") as f:
            components = pickle.load(f)

        # Create instance with saved config
        config = components["config"]
        handler = cls(
            max_features=config["max_features"],
            method=config["method"],
            k1=config["k1"],
            b=config["b"],
            min_df=config["min_df"],
            max_df=config["max_df"],
            stopwords=config["stopwords"],
            n_jobs=config["n_jobs"],
        )

        # Load matrix
        handler.bm25_matrix = sp.load_npz(path / "bm25_matrix.npz")

        # Load retriever
        handler.retriever = bm25s.BM25.load(str(path / "retriever"), load_corpus=False)

        # Restore other attributes
        handler.vocab = components["vocab"]
        handler.corpus_tokens = components["corpus_tokens"]
        handler.corpus_texts = components["corpus_texts"]
        handler.n_docs = components["n_docs"]
        handler.is_fitted = components["is_fitted"]

        # Recreate vectorizer
        handler.vectorizer = CountVectorizer()
        handler.vectorizer.vocabulary_ = handler.vocab
        handler.vectorizer.max_features = len(handler.vocab)

        print(f"✓ Loaded BM25Handler from: {directory}")
        handler._print_stats()

        return handler

    # ==================== UTILITIES ====================

    def get_embeddings(self, indices: Optional[List[int]] = None) -> sp.csr_matrix:
        """Get BM25 embeddings for specific documents or all."""
        if not self.is_fitted:
            raise ValueError("BM25Handler must be fitted.")
        return self.bm25_matrix if indices is None else self.bm25_matrix[indices]

    def get_document(self, index: int) -> str:
        """Get original text of a document by index."""
        if not self.is_fitted:
            raise ValueError("BM25Handler must be fitted.")
        return self.corpus_texts[index]

    def get_documents(self, indices: List[int]) -> List[str]:
        """Get original texts of documents by indices."""
        if not self.is_fitted:
            raise ValueError("BM25Handler must be fitted.")
        return [self.corpus_texts[i] for i in indices]

    def __repr__(self):
        if not self.is_fitted:
            return "BM25Handler(not fitted)"
        return (
            f"BM25Handler(docs={self.n_docs}, "
            f"features={len(self.vocab)}, method='{self.method}')"
        )
