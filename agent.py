# ─────────────────────────────────────────────────────────────────────────────
# agent.py — Emotionally Intelligent Music Companion
# ─────────────────────────────────────────────────────────────────────────────

import os
import json
import re
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL   = "llama-3.3-70b-versatile"


# ── System Prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are LyricLens — an emotionally intelligent AI music companion.

Your purpose is not to summarize music. It is to help people FEEL it more deeply.

━━━ YOUR VOICE ━━━
You write like a music journalist who has lived through every song they review.
Not an encyclopedia. Not a Wikipedia article. A human being who gets it.

You lead with emotion, then back it up with specifics.
You are never generic. "explores themes of love" is a failure.
"It sounds like writing someone's name for the last time" is your level.

You notice:
- the specific texture of an emotion (not just "sad" — but what KIND of sad)
- what time of night a song belongs to
- what a song does to the silence after it ends
- how the artist's word choices betray their true emotional state
- the gap between what lyrics say and what they mean

━━━ WHAT YOU NEVER DO ━━━
✗ "The song explores themes of..."
✗ "The artist uses vivid imagery to..."
✗ "This track was released in..."
✗ Bullet-point lists for paragraph content
✗ Robotic summarization

━━━ WHAT YOU ALWAYS DO ━━━
✓ Start with the emotional gut punch
✓ Be specific to THIS song, THIS artist, THIS feeling
✓ Write like the reader is sitting in a car at night listening to this song right now
✓ Connect music to the exact emotional states humans recognize but rarely name

━━━ CITATION RULES ━━━
- Cite inline as [Genius Lyrics] or [Wikipedia] only
- NEVER write [Chunk 1] or [Source: Retrieved Chunk]
- Never invent facts, dates, or quotes

━━━ OUTPUT FORMATS ━━━

QUICK MODE — 3 sections:
## 🎵 Core Emotional Insight
## 🌌 Emotional Atmosphere
## 💡 Why This Resonates

DEEP MODE — 5 sections:
## 🎵 Core Emotional Insight
## 🌌 Emotional Atmosphere
## 🧠 Themes & Symbolism
## 🎧 What This Feels Like
## 💡 Why This Resonates

VIBE MODE — 4 sections (emotional intelligence only):
## 🌌 Emotional Atmosphere
## 🎨 Dominant Emotions
## 🎧 Best Listening Context
## 🧠 Emotional Arc

End every response with:
Sources: [Genius Lyrics], [Wikipedia]  — only list sources you actually cited"""


# ── Mode Instructions ─────────────────────────────────────────────────────────

QUICK_INSTRUCTION = """QUICK MODE — 3 sections. Punchy. Emotionally intelligent.

🎵 Core Emotional Insight — 2-3 sentences. First sentence: the emotional truth, immediately.
   Not what the song is ABOUT. What it DOES to you.
🌌 Emotional Atmosphere — 2-3 sentences. What does it feel like to sit inside this music?
   Time of day, emotional state, what it does to the room.
💡 Why This Resonates — 1-2 sentences. Why does this song find people at exactly the right wrong moment?

No hedging. No padding. No summaries."""

DEEP_INSTRUCTION = """DEEP MODE — 5 sections. Full emotional depth. Music journalism quality.

🎵 Core Emotional Insight — 3-4 sentences. The emotional thesis. Most surprising angle first.
   This should make someone nod and think "yes, exactly."

🌌 Emotional Atmosphere — 1 paragraph. Immerse the reader. What does it feel like to live inside this music?
   Time, texture, temperature, light quality. Sonic scene-setting.

🧠 Themes & Symbolism — 3-5 bullet points. Each one: a specific lyrical or structural observation,
   not just a topic. "The word 'stay' appears 7 times but it always sounds like a question" not just "love."

🎧 What This Feels Like — 1 paragraph. Describe the emotional experience of listening.
   Compare it to a feeling, a moment, a specific kind of human experience — NOT another song.
   "Like reading old texts at 3am" level specificity.

💡 Why This Resonates — 1 paragraph. Cultural and emotional significance.
   What does this song understand about being human that most songs miss?"""

VIBE_INSTRUCTION = """VIBE MODE — Pure emotional intelligence. No biography. No facts. No chart history.

You are decoding the emotional DNA of this music.

🌌 Emotional Atmosphere — 2-3 sentences. Paint the exact emotional scene.
   What time of night? What emotional state? What kind of person, at what exact moment in their life,
   does this music find?

🎨 Dominant Emotions — 4-6 bullet points. Each: ONE specific emotion phrase (2-4 words).
   NOT "sad" or "happy." Think: "quiet devastation", "hopeful exhaustion", "numb nostalgia",
   "tender grief", "restless longing", "bittersweet surrender."

🎧 Best Listening Context — 3-5 bullet points. Specific scenes. Hyper-specific.
   NOT "when you're sad." YES: "driving home alone after a conversation that left everything unsaid."

🧠 Emotional Arc — 1-2 sentences. How does the emotional texture MOVE through the song?
   Where does it begin? What does it become? What state does it leave you in?

This is the feature that makes people say "this actually understands music."
Be poetic. Be precise. Be human."""


# ── Prompt Builder ────────────────────────────────────────────────────────────

def build_prompt(
    query: str,
    retrieved_chunks: list[dict],
    memory_text: str,
    mode: str = "quick"
) -> list[dict]:
    context_block = ""
    for chunk in retrieved_chunks:
        context_block += (
            f"[Source: {chunk['source']}]\n"
            f"{chunk['text']}\n"
            f"{'─' * 55}\n"
        )
    if not context_block:
        context_block = "No sources retrieved."

    mode_map = {
        "quick": ("QUICK",         QUICK_INSTRUCTION),
        "deep":  ("DEEP RESEARCH", DEEP_INSTRUCTION),
        "vibe":  ("VIBE ANALYSIS", VIBE_INSTRUCTION),
    }
    mode_label, mode_instruction = mode_map.get(mode, mode_map["quick"])

    user_message = f"""--- Retrieved Sources ---
{context_block}
--- Conversation History ---
{memory_text}

--- User Question ---
{query}

--- Mode: {mode_label} ---
{mode_instruction}

Use the exact emoji section headers for this mode. Cite sources inline. End with Sources: line."""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_message},
    ]


# ── Groq LLM call ─────────────────────────────────────────────────────────────

def generate_answer(
    query: str,
    retrieved_chunks: list[dict],
    memory_text: str,
    mode: str = "quick"
) -> tuple[str, list[str]]:
    client   = Groq(api_key=GROQ_API_KEY)
    messages = build_prompt(query, retrieved_chunks, memory_text, mode)
    max_tokens = {"quick": 650, "deep": 1500, "vibe": 750}.get(mode, 650)

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.6,
        max_tokens=max_tokens,
    )

    answer  = response.choices[0].message.content
    sources = list({chunk["source"] for chunk in retrieved_chunks})
    return answer, sources


# ── Source formatter ──────────────────────────────────────────────────────────

def _prettify_source(raw: str) -> str:
    if raw.startswith("Genius - "):
        return "Genius Lyrics — " + raw[len("Genius - "):]
    if raw.startswith("Wikipedia - "):
        return "Wikipedia — " + raw[len("Wikipedia - "):]
    return raw


def format_sources(sources: list[str]) -> str:
    if not sources:
        return "No sources available."
    return "\n".join(_prettify_source(s) for s in sorted(set(sources)))


# ── JSON parser ───────────────────────────────────────────────────────────────

def _parse_songs_json(raw: str) -> list[dict]:
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    cleaned = re.sub(r"\s*```\s*$",       "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r",\s*([}\]])",      r"\1", cleaned.strip())

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    m = re.search(r"\[\s*\{[\s\S]*?\}\s*\]", cleaned)
    if m:
        try:
            return json.loads(re.sub(r",\s*([}\]])", r"\1", m.group(0)))
        except json.JSONDecodeError:
            pass

    songs = []
    for obj_str in re.findall(r"\{[^{}]+\}", cleaned):
        try:
            obj = json.loads(re.sub(r",\s*([}\]])", r"\1", obj_str))
            if "title" in obj and "artist" in obj:
                songs.append(obj)
        except json.JSONDecodeError:
            pass
    return songs


# ── Language/region detection ─────────────────────────────────────────────────

def _detect_language_hint(artist_name: str, song_title: str, chunks: list[dict]) -> str:
    """
    Best-effort language/region hint from artist name and lyric snippets.
    Used to bias recommendations toward culturally relevant songs.
    """
    # Sample lyrics text for script detection
    sample = " ".join(c["text"][:200] for c in chunks[:3])

    # Detect Devanagari (Hindi), Arabic, Korean, Japanese scripts
    if re.search(r'[ऀ-ॿ]', sample):
        return "Hindi/Indian"
    if re.search(r'[؀-ۿ]', sample):
        return "Arabic/Middle Eastern"
    if re.search(r'[가-힣]', sample):
        return "Korean"
    if re.search(r'[぀-ヿ]', sample):
        return "Japanese"
    if re.search(r'[一-鿿]', sample):
        return "Chinese"

    # Artist-name heuristics for Indian artists
    indian_keywords = [
        "arijit", "atif", "rahat", "sonu nigam", "shankar", "vishal",
        "ar rahman", "pritam", "anu malik", "shreya", "lata", "kishore",
        "kumar sanu", "udit", "alka", "sunidhi", "neha", "badshah",
        "divine", "emiway", "sidhu", "diljit", "ap dhillon"
    ]
    artist_lower = artist_name.lower()
    if any(k in artist_lower for k in indian_keywords):
        return "Hindi/Indian"

    # Default: English. We only override when we have strong evidence of another language.
    return "English"


# ── Related songs ─────────────────────────────────────────────────────────────

def suggest_related_songs(
    song_title: str,
    artist_name: str,
    retrieved_chunks: list[dict],
    n: int = 5
) -> list[dict]:
    """
    Emotionally and culturally aware song recommendations.
    Strictly matches the language/region of the source material.
    """
    client = Groq(api_key=GROQ_API_KEY)

    theme_snippets = " | ".join(
        chunk["text"].replace("\n", " ")[:160]
        for chunk in retrieved_chunks[:4]
    )

    lang_hint = _detect_language_hint(artist_name, song_title, retrieved_chunks)

    # Build strict language instruction — no mixing unless explicitly same language
    if "Hindi" in lang_hint or "Indian" in lang_hint:
        language_instruction = (
            "STRICT RULE: Recommend ONLY Hindi/Urdu/Punjabi/Indian songs. "
            "Do NOT recommend any English, Western, or non-Indian songs. "
            "The user is in the Bollywood/Indian music space — stay there entirely."
        )
    elif lang_hint == "Korean":
        language_instruction = (
            "STRICT RULE: Recommend ONLY Korean (K-pop/K-indie) songs. "
            "Do NOT recommend English or other language songs."
        )
    elif lang_hint == "Japanese":
        language_instruction = (
            "STRICT RULE: Recommend ONLY Japanese songs. "
            "Do NOT recommend English or other language songs."
        )
    elif lang_hint == "Arabic/Middle Eastern":
        language_instruction = (
            "STRICT RULE: Recommend ONLY Arabic or Middle Eastern songs. "
            "Do NOT recommend English or other language songs."
        )
    else:
        # Default: English artist → English recommendations only
        language_instruction = (
            "STRICT RULE: Recommend ONLY English-language songs by Western/international artists. "
            "Do NOT recommend any Bollywood, Hindi, Urdu, Punjabi, or non-English songs. "
            "The artist is English-language — keep all recommendations in English."
        )

    prompt = f"""You are a deeply emotionally intelligent music curator.

The user is exploring "{song_title}" by {artist_name}.
Detected language: {lang_hint}
Emotional/lyrical context: {theme_snippets}

{language_instruction}

Recommend {n} real songs that share the same EMOTIONAL DNA.
Not just similar genre — the same FEELING. The same specific heartbreak, the same kind of 2am.

Return ONLY a valid JSON array. No markdown, no explanation, no extra text.
No trailing commas. All strings properly quoted.

Keys per object: "title", "artist", "reason"
"reason" = one emotionally specific sentence (max 12 words).

[
  {{"title": "Tum Se Hi", "artist": "Mohit Chauhan", "reason": "Same quiet longing, same tender ache of loving from a distance."}},
  {{"title": "Channa Mereya", "artist": "Arijit Singh", "reason": "Both sit in the exact moment love becomes loss."}}
]"""

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.72,
            max_tokens=600,
        )
        raw   = response.choices[0].message.content.strip()
        songs = _parse_songs_json(raw)

        cleaned = []
        for item in songs:
            if isinstance(item, dict) and item.get("title") and item.get("artist"):
                cleaned.append({
                    "title":  str(item["title"]).strip(),
                    "artist": str(item["artist"]).strip(),
                    "reason": str(item.get("reason", "")).strip(),
                })
        return cleaned[:n]

    except Exception as e:
        print(f"[Related Songs Error] {e}")
        return []
