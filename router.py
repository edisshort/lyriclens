# ─────────────────────────────────────────────────────────────────────────────
# router.py — Query Routing Agent
#
# A lightweight Groq call that classifies every user query into a retrieval
# strategy BEFORE we hit ChromaDB. This is the "agentic" behavior:
#
#   User asks → Agent decides HOW to search → Retrieval tuned to intent
#
# QUERY TYPES:
#   emotional    → "what does this feel like", "mood", "vibe"
#                  → high vector weight, lyrics priority, more chunks
#   factual      → "when was", "who produced", "release date", "chart"
#                  → high BM25 weight, Wikipedia priority, fewer chunks
#   atmospheric  → "atmosphere", "listening context", "night drive"
#                  → max vector weight, lyrics priority
#   thematic     → "themes", "meaning", "symbolism", "about"
#                  → balanced weights, both sources
#   biographical → "career", "background", "history", "influences"
#                  → Wikipedia priority, BM25-heavy
#
# ─────────────────────────────────────────────────────────────────────────────

import os
import json
import re
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL   = "llama-3.3-70b-versatile"

# ── Retrieval strategy presets ────────────────────────────────────────────────
STRATEGIES = {
    "emotional": {
        "bm25_weight":    0.25,
        "vector_weight":  0.75,
        "top_k":          7,
        "source_priority": "lyrics",
        "description":    "High semantic weight, lyrics-first — reading the emotional subtext",
    },
    "factual": {
        "bm25_weight":    0.65,
        "vector_weight":  0.35,
        "top_k":          4,
        "source_priority": "wikipedia",
        "description":    "Keyword-heavy, Wikipedia-first — surface-level facts need precision",
    },
    "atmospheric": {
        "bm25_weight":    0.15,
        "vector_weight":  0.85,
        "top_k":          6,
        "source_priority": "lyrics",
        "description":    "Maximum semantic, lyrics-only — capturing sonic atmosphere",
    },
    "thematic": {
        "bm25_weight":    0.40,
        "vector_weight":  0.60,
        "top_k":          7,
        "source_priority": "both",
        "description":    "Balanced — themes live in both lyrics and context",
    },
    "biographical": {
        "bm25_weight":    0.60,
        "vector_weight":  0.40,
        "top_k":          5,
        "source_priority": "wikipedia",
        "description":    "Wikipedia-first — artist history and background",
    },
    "recommendation": {
        "bm25_weight":    0.20,
        "vector_weight":  0.80,
        "top_k":          6,
        "source_priority": "lyrics",
        "description":    "Semantic-heavy — finding emotional similarity for recommendations",
    },
}

DEFAULT_STRATEGY = "thematic"


# ── Fast rule-based fallback (no API call needed) ─────────────────────────────

_FACTUAL_PATTERNS = [
    r'\b(when|year|date|released?|chart|billboard|grammy|award|label|producer|produced|album|sample|collaborated?|certified|sales|streams)\b',
    r'\bwho (is|was|made|produced|wrote)\b',
]
_EMOTIONAL_PATTERNS = [
    r'\b(feel|feeling|emotion|emotional|vibe|vibes|mood|sad|lonely|happy|heartbreak|nostalgia|intimate|vulnerable)\b',
    r'\b(what does|what.?s it like|how does|what kind of|what.?s the feeling)\b',
]
_ATMOSPHERIC_PATTERNS = [
    r'\b(atmosphere|atmospheric|cinematic|sonic|listening context|time of night|best listened|listening experience)\b',
    r'\b(night drive|midnight|rainy|late night|dark room|alone)\b',
]
_BIOGRAPHICAL_PATTERNS = [
    r'\b(career|history|background|influenced|grew up|early life|discography|journey|evolution|before|debut)\b',
]
_RECOMMENDATION_PATTERNS = [
    r'\b(similar|recommend|like this|other songs|suggest|also listen|related|if i like)\b',
]

def _rule_based_route(query: str) -> str | None:
    q = query.lower()
    if any(re.search(p, q) for p in _FACTUAL_PATTERNS):     return "factual"
    if any(re.search(p, q) for p in _BIOGRAPHICAL_PATTERNS): return "biographical"
    if any(re.search(p, q) for p in _ATMOSPHERIC_PATTERNS):  return "atmospheric"
    if any(re.search(p, q) for p in _EMOTIONAL_PATTERNS):    return "emotional"
    if any(re.search(p, q) for p in _RECOMMENDATION_PATTERNS): return "recommendation"
    return None   # uncertain → use LLM


# ── LLM-based classification ──────────────────────────────────────────────────

ROUTER_PROMPT = """You are a query routing agent for a music intelligence platform.

Classify this music query into EXACTLY ONE category:
- emotional     → about feelings, emotions, vibe, mood, how music makes you feel
- factual       → concrete facts: release dates, charts, producers, collaborations, certifications
- atmospheric   → atmosphere, sonic texture, listening context, time/place to listen
- thematic      → themes, symbolism, meaning, metaphors, lyrical interpretation
- biographical  → artist history, career, background, influences, evolution
- recommendation → similar songs, what else to listen to, comparisons

Return ONLY a JSON object with one key: {"type": "..."}
No explanation. No markdown. Just JSON."""


def _llm_route(query: str) -> str:
    client = Groq(api_key=GROQ_API_KEY)
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": ROUTER_PROMPT},
                {"role": "user",   "content": f"Query: {query}"},
            ],
            temperature=0.0,   # deterministic classification
            max_tokens=30,     # very small — just needs {"type": "..."}
        )
        raw = response.choices[0].message.content.strip()
        # strip markdown fences
        raw = re.sub(r"```(?:json)?|```", "", raw).strip()
        result = json.loads(raw)
        query_type = result.get("type", DEFAULT_STRATEGY)
        if query_type in STRATEGIES:
            return query_type
        return DEFAULT_STRATEGY
    except Exception as e:
        print(f"[Router LLM Error] {e} — using fallback")
        return DEFAULT_STRATEGY


# ── Public API ────────────────────────────────────────────────────────────────

def route_query(query: str) -> dict:
    """
    Analyze the user query and return the optimal retrieval strategy.

    Steps:
      1. Try fast rule-based routing (no API call, ~0ms)
      2. If uncertain, call Groq for classification (~200ms, 30 tokens)
      3. Return the matching strategy preset

    Returns:
        dict with keys: bm25_weight, vector_weight, top_k, source_priority,
                        query_type, description
    """
    # Step 1: rule-based (fast path)
    query_type = _rule_based_route(query)

    # Step 2: LLM if rule-based is unsure
    if query_type is None:
        query_type = _llm_route(query)

    strategy = STRATEGIES.get(query_type, STRATEGIES[DEFAULT_STRATEGY]).copy()
    strategy["query_type"] = query_type

    print(f"[Router] '{query[:60]}...' → {query_type} | "
          f"BM25={strategy['bm25_weight']} VEC={strategy['vector_weight']} "
          f"top_k={strategy['top_k']} src={strategy['source_priority']}")

    return strategy
