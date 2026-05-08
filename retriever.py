# ─────────────────────────────────────────────────────────────────────────────
# retriever.py — Hybrid BM25 + Vector Search
# ─────────────────────────────────────────────────────────────────────────────
#
# This is the RETRIEVAL BRAIN of LyricLens — it finds the most relevant
# text chunks for a given user query using TWO different search strategies
# combined together (Hybrid RAG).
#
# ── Why Hybrid? ──────────────────────────────────────────────────────────────
#
#  BM25 (keyword search):
#    • Great at exact matches: "Blinding Lights", "Kendrick Lamar", "DAMN."
#    • Fails when the query is abstract: "songs about emptiness and hope"
#
#  Vector Search (semantic search):
#    • Great at meaning/vibe queries: finds "Empty" when you ask about loneliness
#    • Can miss exact song/artist name matches if phrased differently
#
#  Hybrid = best of both worlds
#    final_score = 0.4 × BM25_score + 0.6 × vector_score
#    (We weight vector slightly higher because LyricLens users often ask
#     semantic questions about themes, emotions, and metaphors)
#
# ── Flow ──────────────────────────────────────────────────────────────────────
#
#  1. Build BM25 index from all chunks (in-memory, per session)
#  2. Run BM25 search → top 10 results with scores
#  3. Run ChromaDB vector search → top 10 results with distances
#  4. Normalise both score lists to [0, 1]
#  5. Merge by chunk text, combine scores with 0.4/0.6 weighting
#  6. Sort by final score, deduplicate, return top-k chunks
#
# ─────────────────────────────────────────────────────────────────────────────

import re
import chromadb
from rank_bm25 import BM25Okapi   # BM25 implementation


# ── Constants ─────────────────────────────────────────────────────────────────

BM25_WEIGHT    = 0.4   # weight for keyword score
VECTOR_WEIGHT  = 0.6   # weight for semantic score
CANDIDATE_K    = 10    # how many candidates to pull from each retriever before merging


# ── Tokeniser (shared by BM25) ────────────────────────────────────────────────

def _tokenise(text: str) -> list[str]:
    """
    Simple whitespace + punctuation tokeniser for BM25.
    Lowercase, split on non-word characters, remove empty tokens.

    BM25 works on token lists, not raw strings — this converts text to tokens.
    """
    tokens = re.split(r"\W+", text.lower())
    return [t for t in tokens if t]


# ── BM25 Index ────────────────────────────────────────────────────────────────

class BM25Index:
    """
    Wraps rank_bm25.BM25Okapi to add score normalisation and chunk lookup.

    BM25Okapi is the "Okapi BM25" variant — a probabilistic ranking function
    used by search engines like Elasticsearch. It scores documents based on:
      - Term frequency in the document (TF)
      - Inverse document frequency across the corpus (IDF)
      - Document length normalisation (prevents long docs from always winning)
    """

    def __init__(self, chunks: list[str]):
        """
        Build the BM25 index from a list of text chunks.

        Args:
            chunks: all text chunks (lyrics + article paragraphs)
        """
        self.chunks = chunks
        tokenised   = [_tokenise(chunk) for chunk in chunks]
        self.bm25   = BM25Okapi(tokenised)

    def search(self, query: str, top_k: int = CANDIDATE_K) -> list[tuple[str, float]]:
        """
        Search the BM25 index for the most relevant chunks.

        Args:
            query : user's natural language query
            top_k : number of results to return

        Returns:
            List of (chunk_text, raw_bm25_score) tuples, sorted by score desc.
        """
        query_tokens = _tokenise(query)
        scores       = self.bm25.get_scores(query_tokens)   # array of scores, one per chunk

        # Pair each chunk with its score, sort descending
        scored = sorted(
            zip(self.chunks, scores),
            key=lambda x: x[1],
            reverse=True
        )

        return scored[:top_k]


# ── Normalisation ─────────────────────────────────────────────────────────────

def _normalise_scores(scored_list: list[tuple[str, float]]) -> dict[str, float]:
    """
    Min-max normalise a list of (text, score) tuples into a dict {text: 0..1}.

    Why normalise? BM25 scores are unbounded floats (could be 3.7 or 0.002).
    ChromaDB distances are also different units (cosine distance 0..2).
    We can't combine them directly — normalisation puts both on a 0-1 scale.

    Args:
        scored_list: list of (text, score) — higher score = better match

    Returns:
        Dict mapping chunk text → normalised score in [0, 1]
    """
    if not scored_list:
        return {}

    scores = [s for _, s in scored_list]
    min_s, max_s = min(scores), max(scores)

    # Avoid division by zero if all scores are identical
    score_range = max_s - min_s if max_s != min_s else 1.0

    return {
        text: (score - min_s) / score_range
        for text, score in scored_list
    }


# ── Vector Search ─────────────────────────────────────────────────────────────

def vector_search(
    query: str,
    collection: chromadb.Collection,
    top_k: int = CANDIDATE_K
) -> list[tuple[str, float, str]]:
    """
    Query ChromaDB for semantically similar chunks using cosine similarity.

    ChromaDB returns distances (lower = more similar), so we convert to
    similarity scores (higher = more similar) by: similarity = 1 - distance.

    Args:
        query      : user query string — ChromaDB will embed it automatically
        collection : the ChromaDB collection built during ingest
        top_k      : number of results to retrieve

    Returns:
        List of (chunk_text, similarity_score, source_label) tuples.
    """
    results = collection.query(
        query_texts=[query],
        n_results=min(top_k, collection.count()),   # can't request more than exists
        include=["documents", "distances", "metadatas"]
    )

    chunks    = results["documents"][0]       # list of matched chunk texts
    distances = results["distances"][0]       # cosine distances (lower = closer)
    metadatas = results["metadatas"][0]       # list of {source: ...} dicts

    # Convert distance → similarity; cosine distance ∈ [0, 2] → similarity ∈ [-1, 1]
    # We clamp to [0, 1] for safety
    output = []
    for chunk, dist, meta in zip(chunks, distances, metadatas):
        similarity = max(0.0, 1.0 - dist)
        source     = meta.get("source", "Unknown")
        output.append((chunk, similarity, source))

    return output


# ── Source Lookup for BM25-only Results ──────────────────────────────────────

def _lookup_sources_from_chroma(
    chunks: list[str],
    collection: chromadb.Collection
) -> dict[str, str]:
    """
    For chunks that only came from BM25 (not vector search), query ChromaDB
    to retrieve their stored source metadata.

    Why this is needed:
      BM25 works on raw text — it has no knowledge of where each chunk came
      from. ChromaDB stores that metadata. So we do a small targeted query
      for each BM25-only chunk using the first ~60 chars as a substring filter.

    Args:
        chunks     : list of chunk texts that need source lookup
        collection : the ChromaDB collection containing metadata

    Returns:
        Dict of {chunk_text: source_label}
    """
    result_map: dict[str, str] = {}

    for chunk_text in chunks:
        try:
            # Use first 60 non-newline characters as a reliable substring filter
            snippet = chunk_text.replace("\n", " ")[:60].strip()
            if not snippet:
                result_map[chunk_text] = "Unknown Source"
                continue

            hits = collection.get(
                where_document={"$contains": snippet},
                include=["documents", "metadatas"]
            )

            # Find the exact matching document among hits
            found = False
            for doc, meta in zip(hits["documents"], hits["metadatas"]):
                if doc == chunk_text:
                    result_map[chunk_text] = meta.get("source", "Unknown Source")
                    found = True
                    break

            if not found:
                # Fallback: take the first hit's source if snippet matched something
                if hits["documents"]:
                    result_map[chunk_text] = hits["metadatas"][0].get("source", "Unknown Source")
                else:
                    result_map[chunk_text] = "Unknown Source"

        except Exception:
            result_map[chunk_text] = "Unknown Source"

    return result_map


# ── Hybrid Merge ──────────────────────────────────────────────────────────────

def hybrid_retrieve(
    query: str,
    all_chunks: list[str],
    collection: chromadb.Collection,
    top_k: int = 5,
    bm25_weight: float = BM25_WEIGHT,
    vector_weight: float = VECTOR_WEIGHT,
    source_priority: str = "both",   # "lyrics" | "wikipedia" | "both"
) -> list[dict]:
    """
    Combine BM25 and vector search with routing-aware weights and source filtering.

    Args:
        query          : user's question (may be context-expanded)
        all_chunks     : all ingested text chunks
        collection     : ChromaDB collection
        top_k          : final number of chunks to return
        bm25_weight    : weight for keyword score (from router)
        vector_weight  : weight for semantic score (from router)
        source_priority: "lyrics" → prefer Genius chunks
                         "wikipedia" → prefer Wikipedia chunks
                         "both" → no filter (default)

    Returns:
        List of dicts with keys: text, source, score
    """
    if not all_chunks:
        return []

    # ── Filter chunks by source priority ──────────────────────────────────
    # When the router says "lyrics" or "wikipedia", we narrow the BM25 corpus
    # to that source type. This focuses retrieval on what actually matters.
    if source_priority == "lyrics":
        search_chunks = [c for c in all_chunks if not c.startswith("Wikipedia")]
        # Fallback: if filtering removed everything, use all chunks
        if not search_chunks:
            search_chunks = all_chunks
    elif source_priority == "wikipedia":
        search_chunks = [c for c in all_chunks if not c.startswith("Genius")]
        if not search_chunks:
            search_chunks = all_chunks
    else:
        search_chunks = all_chunks

    # ── BM25 search ───────────────────────────────────────────────────────
    bm25_index   = BM25Index(search_chunks)
    bm25_results = bm25_index.search(query, top_k=CANDIDATE_K)
    bm25_normed  = _normalise_scores(bm25_results)

    # ── Vector search ─────────────────────────────────────────────────────
    vec_results = vector_search(query, collection, top_k=CANDIDATE_K)

    source_map: dict[str, str]   = {}
    vec_normed: dict[str, float] = {}

    for chunk, sim, source in vec_results:
        # Apply source filter to vector results too
        if source_priority == "lyrics"    and source.startswith("Wikipedia"):
            continue
        if source_priority == "wikipedia" and source.startswith("Genius"):
            continue
        vec_normed[chunk] = sim
        source_map[chunk] = source

    # Source lookup for BM25-only chunks
    bm25_only = [chunk for chunk, _ in bm25_results if chunk not in source_map]
    if bm25_only:
        source_map.update(_lookup_sources_from_chroma(bm25_only, collection))

    # ── Merge with routing-supplied weights ───────────────────────────────
    all_candidate_texts = set(bm25_normed.keys()) | set(vec_normed.keys())

    merged = []
    for chunk_text in all_candidate_texts:
        b_score = bm25_normed.get(chunk_text, 0.0)
        v_score = vec_normed.get(chunk_text, 0.0)
        final   = bm25_weight * b_score + vector_weight * v_score
        merged.append({
            "text":   chunk_text,
            "source": source_map.get(chunk_text, "Unknown"),
            "score":  round(final, 4),
        })

    merged.sort(key=lambda x: x["score"], reverse=True)
    return merged[:top_k]
