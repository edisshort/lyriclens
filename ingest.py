# ─────────────────────────────────────────────────────────────────────────────
# ingest.py — Fetch · Chunk · Embed · Store
# ─────────────────────────────────────────────────────────────────────────────
#
#  Step 1 — FETCH
#    • Genius REST API  → song metadata (title, artist, cover art, etc.)
#      Uses the token-authenticated API directly — no scraping, no Cloudflare.
#    • lyrics.ovh       → raw lyrics (free, no auth, works everywhere)
#    • Deezer API       → high-res cover art (free, no auth)
#    • Wikipedia        → artist / song background article
#
#  Step 2 — CHUNK
#    • Lyrics  → groups of 8 lines
#    • Articles → paragraphs (max 1500 chars)
#
#  Step 3 + 4 — EMBED & STORE
#    • ChromaDB DefaultEmbeddingFunction (ONNX, ~120MB, no PyTorch)
#
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import os
import re
import json
import shutil
import string
import urllib.request
import urllib.parse
import wikipediaapi
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()

# ── Constants ─────────────────────────────────────────────────────────────────

GENIUS_TOKEN      = os.getenv("GENIUS_ACCESS_TOKEN")
GENIUS_API_BASE   = "https://api.genius.com"
CHROMA_PATH       = "./chroma_db"
COLLECTION_NAME   = "lyriclens_chunks"
LYRIC_CHUNK_LINES = 8
WIKI_USER_AGENT   = "LyricLens/1.0 (music research app)"

_HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/124 Safari/537.36"
    )
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get(url: str, *, headers: dict | None = None, timeout: int = 10) -> dict | None:
    """Simple JSON GET — returns parsed dict or None on any error."""
    h = {**_HTTP_HEADERS, **(headers or {})}
    try:
        req = urllib.request.Request(url, headers=h)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except Exception as e:
        print(f"[HTTP] GET {url[:80]}… failed: {e}")
        return None


def _title_case(s: str) -> str:
    """'frank ocean' → 'Frank Ocean'"""
    return string.capwords(s)


# ── Cover art — Deezer → iTunes ───────────────────────────────────────────────

def fetch_cover_art(song_title: str, artist_name: str) -> str | None:
    """
    Try Deezer first (free, no key, massive catalog).
    Falls back to iTunes if Deezer returns nothing.
    """
    # 1. Deezer
    for q in [f"{song_title} {artist_name}", song_title]:
        data = _get(
            f"https://api.deezer.com/search?q={urllib.parse.quote(q)}&limit=10"
        )
        if not data:
            break
        for track in data.get("data", []):
            found = track.get("artist", {}).get("name", "").lower()
            if artist_name.lower() not in found and found not in artist_name.lower():
                continue
            art = (
                track.get("album", {}).get("cover_xl")
                or track.get("album", {}).get("cover_big")
                or track.get("album", {}).get("cover_medium")
            )
            if art and art.startswith("http"):
                print(f"[Deezer] Cover: {art}")
                return art

    # 2. iTunes fallback
    data = _get(
        f"https://itunes.apple.com/search"
        f"?term={urllib.parse.quote(song_title + ' ' + artist_name)}"
        f"&media=music&entity=song&limit=5"
    )
    if data:
        for r in data.get("results", []):
            art = r.get("artworkUrl100", "")
            if art:
                art = art.replace("100x100bb", "600x600bb")
                print(f"[iTunes] Cover: {art}")
                return art

    print(f"[Cover] Not found for '{song_title}' by '{artist_name}'")
    return None


# ── Genius REST API — metadata only (no scraping) ────────────────────────────

def fetch_genius_metadata(song_title: str, artist_name: str) -> dict:
    """
    Call the Genius search API with the Bearer token.
    This is a proper API call — not scraping — so Cloudflare never blocks it.
    Returns a metadata dict (cover_art may be None; we fill it from Deezer).
    """
    if not GENIUS_TOKEN:
        print("[Genius] No token — skipping metadata")
        return {"title": song_title, "artist": artist_name}

    auth = {"Authorization": f"Bearer {GENIUS_TOKEN}"}
    data = _get(
        f"{GENIUS_API_BASE}/search?q={urllib.parse.quote(song_title + ' ' + artist_name)}",
        headers=auth,
    )
    if not data:
        return {"title": song_title, "artist": artist_name}

    hits = data.get("response", {}).get("hits", [])
    for hit in hits:
        result = hit.get("result", {})
        pa     = result.get("primary_artist", {})
        return {
            "title":        result.get("title", song_title),
            "artist":       result.get("artist_names") or pa.get("name", artist_name),
            "album":        None,          # not in basic search results
            "release_date": result.get("release_date_for_display"),
            "cover_art":    (
                result.get("song_art_image_url")
                or result.get("song_art_image_thumbnail_url")
            ),
            "artist_image": pa.get("image_url"),
            "genius_url":   result.get("url"),
        }

    return {"title": song_title, "artist": artist_name}


# ── Lyrics — lyrics.ovh (free, no auth, no scraping) ─────────────────────────

def fetch_lyrics(song_title: str, artist_name: str) -> str | None:
    """
    Fetch lyrics from lyrics.ovh.
    Tries both the given name and a title-cased version.
    """
    for artist in [artist_name, _title_case(artist_name)]:
        for title in [song_title, _title_case(song_title)]:
            url  = (
                f"https://api.lyrics.ovh/v1/"
                f"{urllib.parse.quote(artist)}/{urllib.parse.quote(title)}"
            )
            data = _get(url, timeout=12)
            if data and data.get("lyrics"):
                print(f"[Lyrics.ovh] Found lyrics for '{title}' by '{artist}'")
                return data["lyrics"]

    print(f"[Lyrics.ovh] Not found: '{song_title}' by '{artist_name}'")
    return None


# ── Combined fetch ────────────────────────────────────────────────────────────

def fetch_song_and_lyrics(
    song_title: str, artist_name: str
) -> tuple[dict, str | None]:
    """
    Returns (song_meta dict, lyrics string or None).
    Uses Genius API for metadata, Deezer/iTunes for cover art, lyrics.ovh for lyrics.
    All three are proper API calls — no scraping, no Cloudflare blocks.
    """
    print(f"[Ingest] Fetching metadata: '{song_title}' by {artist_name}…")
    meta   = fetch_genius_metadata(song_title, artist_name)

    # Cover art: Deezer/iTunes first, then Genius URL if available
    cover  = fetch_cover_art(song_title, artist_name)
    if not cover:
        cover = meta.get("cover_art")
    meta["cover_art"] = cover
    print(f"[Ingest] Cover art: {cover or 'not found'}")

    print(f"[Ingest] Fetching lyrics via lyrics.ovh…")
    lyrics = fetch_lyrics(song_title, artist_name)

    return meta, lyrics


# ── Wikipedia ─────────────────────────────────────────────────────────────────

def fetch_wikipedia_article(search_term: str) -> str | None:
    """
    Fetch Wikipedia article text.
    Tries the term as-is, then title-cased (handles 'frank ocean' → 'Frank Ocean').
    """
    wiki = wikipediaapi.Wikipedia(language="en", user_agent=WIKI_USER_AGENT)
    for term in [search_term, _title_case(search_term)]:
        page = wiki.page(term)
        if page.exists():
            print(f"[Wikipedia] Found article: '{term}'")
            return page.text
    print(f"[Wikipedia] No article for '{search_term}'")
    return None


# ── ChromaDB ──────────────────────────────────────────────────────────────────

def _get_chroma_collection(fresh: bool = False) -> chromadb.Collection:
    if fresh and os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        print(f"[ChromaDB] Wiped old DB — starting fresh")

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    ef     = embedding_functions.DefaultEmbeddingFunction()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )


# ── Step 2: Chunk ─────────────────────────────────────────────────────────────

def chunk_lyrics(lyrics: str, lines_per_chunk: int = LYRIC_CHUNK_LINES) -> list[str]:
    lyrics = re.sub(r"^\d+\s*Contributors?.*\n", "", lyrics, flags=re.IGNORECASE)
    lyrics = re.sub(r"\d*Embed$", "", lyrics).strip()
    lines  = [l.strip() for l in lyrics.split("\n") if l.strip()]
    return [
        "\n".join(lines[i: i + lines_per_chunk])
        for i in range(0, len(lines), lines_per_chunk)
        if lines[i: i + lines_per_chunk]
    ]


def chunk_article(article_text: str, max_chars: int = 1500) -> list[str]:
    paragraphs = [p.strip() for p in article_text.split("\n\n") if p.strip()]
    chunks = []
    for para in paragraphs:
        if len(para) <= max_chars:
            chunks.append(para)
        else:
            sentences = re.split(r"(?<=[.!?])\s+", para)
            current   = ""
            for s in sentences:
                if len(current) + len(s) <= max_chars:
                    current += (" " if current else "") + s
                else:
                    if current:
                        chunks.append(current)
                    current = s
            if current:
                chunks.append(current)
    return chunks


# ── Step 3+4: Embed & Store ───────────────────────────────────────────────────

def store_chunks(
    chunks: list[str],
    source_label: str,
    collection: chromadb.Collection,
) -> None:
    if not chunks:
        return
    collection.upsert(
        documents=chunks,
        ids=[f"{source_label}-{i}" for i in range(len(chunks))],
        metadatas=[{"source": source_label} for _ in chunks],
    )
    print(f"[ChromaDB] Stored {len(chunks)} chunks from '{source_label}'")


# ── Main Public Function ───────────────────────────────────────────────────────

def ingest(
    song_title: str,
    artist_name: str,
) -> tuple[list[str], list[str], chromadb.Collection, dict]:
    """
    Full pipeline: fetch → chunk → embed → store.
    Returns (lyric_chunks, article_chunks, collection, song_meta).
    """
    collection     = _get_chroma_collection(fresh=True)
    lyric_chunks   = []
    article_chunks = []

    # ── Metadata + Lyrics ────────────────────────────────────────────────
    song_meta, lyrics = fetch_song_and_lyrics(song_title, artist_name)

    if lyrics:
        lyric_chunks = chunk_lyrics(lyrics)
        store_chunks(lyric_chunks, f"Genius - {song_title} by {artist_name}", collection)
    else:
        print(f"[Ingest] No lyrics found for '{song_title}'")

    # ── Wikipedia — Artist ────────────────────────────────────────────────
    print(f"[Ingest] Fetching Wikipedia: {artist_name}…")
    artist_article = fetch_wikipedia_article(artist_name)
    if artist_article:
        ac = chunk_article(artist_article)
        store_chunks(ac, f"Wikipedia - {artist_name}", collection)
        article_chunks.extend(ac)

    # ── Wikipedia — Song ─────────────────────────────────────────────────
    print(f"[Ingest] Fetching Wikipedia: {song_title}…")
    song_article = fetch_wikipedia_article(f"{song_title} ({artist_name} song)")
    if not song_article:
        song_article = fetch_wikipedia_article(song_title)
    if song_article:
        sc = chunk_article(song_article)
        store_chunks(sc, f"Wikipedia - {song_title}", collection)
        article_chunks.extend(sc)

    return lyric_chunks, article_chunks, collection, song_meta
