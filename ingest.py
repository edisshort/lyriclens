# ─────────────────────────────────────────────────────────────────────────────
# ingest.py — Fetch · Chunk · Embed · Store
# ─────────────────────────────────────────────────────────────────────────────
#
# This file is the DATA PIPELINE for LyricLens. Every time a user asks about
# a song or artist, we:
#
#  Step 1 — FETCH
#    • Genius API  → raw lyrics for the requested song
#    • Wikipedia   → artist/album background article
#
#  Step 2 — CHUNK
#    • Lyrics  → split into groups of lines (e.g. every 8 lines = 1 chunk)
#    • Articles → split into paragraphs
#    Different chunking strategies are used because lyrics have very short
#    lines (splitting by paragraph would give giant chunks), while Wikipedia
#    articles are naturally paragraph-structured.
#
#  Step 3 — EMBED
#    • Each chunk is converted to a dense vector using sentence-transformers
#      model "all-MiniLM-L6-v2" (fast, 384-dim, good semantic quality)
#
#  Step 4 — STORE
#    • Vectors + raw text are saved in ChromaDB (local, on-disk vector DB)
#    • We also return the raw chunks list so retriever.py can build a BM25 index
#
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import os
import re
import json
import shutil
import urllib.request
import urllib.parse
import lyricsgenius
import wikipediaapi
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

# Load API keys from .env
load_dotenv()

# ── Constants ─────────────────────────────────────────────────────────────────

GENIUS_TOKEN   = os.getenv("GENIUS_ACCESS_TOKEN")
CHROMA_PATH    = "./chroma_db"          # where ChromaDB saves data on disk
COLLECTION_NAME = "lyriclens_chunks"   # name of our vector collection
EMBED_MODEL    = "all-MiniLM-L6-v2"   # sentence-transformers model
LYRIC_CHUNK_LINES = 8                  # how many lyric lines per chunk
WIKI_USER_AGENT   = "LyricLens/1.0 (music research app)"


# ── Cover art lookup (Deezer → iTunes → Genius fallback) ─────────────────────

def fetch_deezer_cover(song_title: str, artist_name: str) -> str | None:
    """
    Query Deezer's free public API for album cover art.
    No API key required. Returns a 1000×1000 JPEG URL.
    """
    try:
        query = urllib.parse.quote(f'artist:"{artist_name}" track:"{song_title}"')
        url   = f"https://api.deezer.com/search?q={query}&limit=5"
        req   = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read())
        for track in data.get("data", []):
            art = track.get("album", {}).get("cover_xl") or track.get("album", {}).get("cover_big")
            if art and art.startswith("http"):
                print(f"[Deezer] Cover found: {art}")
                return art
    except Exception as e:
        print(f"[Deezer] Cover fetch failed: {e}")

    # Fallback: iTunes
    try:
        query = urllib.parse.quote(f"{song_title} {artist_name}")
        url   = f"https://itunes.apple.com/search?term={query}&media=music&entity=song&limit=5"
        req   = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=6) as resp:
            data = json.loads(resp.read())
        for result in data.get("results", []):
            art = result.get("artworkUrl100", "")
            if art:
                print(f"[iTunes] Cover found: {art}")
                return art.replace("100x100bb", "600x600bb")
    except Exception as e:
        print(f"[iTunes] Cover fetch failed: {e}")

    return None


# ── Clients / Models (initialised once, reused across calls) ──────────────────

def _get_genius_client() -> lyricsgenius.Genius:
    """Create a Genius API client with timeouts and verbose logging off."""
    genius = lyricsgenius.Genius(GENIUS_TOKEN)
    genius.verbose = False        # suppress print statements
    genius.remove_section_headers = True   # strip [Chorus], [Verse] tags
    genius.timeout = 15
    return genius


def _get_chroma_collection(fresh: bool = False) -> chromadb.Collection:
    """
    Open (or create) a persistent ChromaDB collection.

    Args:
        fresh : if True, wipe the entire chroma_db directory first.
                This is the only safe way to reset when the on-disk
                metadata may be from an incompatible ChromaDB version.
    """
    if fresh and os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        print(f"[ChromaDB] Wiped old DB at '{CHROMA_PATH}' — starting fresh")

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    # DefaultEmbeddingFunction uses ONNX runtime — same model (all-MiniLM-L6-v2)
    # but ~120MB instead of ~500MB (no PyTorch required)
    ef = embedding_functions.DefaultEmbeddingFunction()
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


# ── Step 1: Fetch ─────────────────────────────────────────────────────────────

def fetch_lyrics(song_title: str, artist_name: str) -> str | None:
    """
    Fetch raw lyrics from the Genius API.

    Returns:
        A string of lyrics, or None if not found.
    """
    genius = _get_genius_client()
    try:
        song = genius.search_song(song_title, artist_name)
        if song:
            return getattr(song, "lyrics", None)
        return None
    except Exception as e:
        print(f"[Genius Error] {e}")
        return None


def fetch_song_and_lyrics(song_title: str, artist_name: str) -> tuple[dict, str | None]:
    """
    Single Genius API call that returns BOTH metadata and lyrics.
    Avoids the double-search that fetch_song_metadata + fetch_lyrics would do.

    Returns:
        (meta_dict, lyrics_string_or_None)
    """
    genius = _get_genius_client()
    try:
        song = genius.search_song(song_title, artist_name)
        if not song:
            return {"title": song_title, "artist": artist_name}, None

        def _artist_str(s):
            for attr in ("artist_names", "artist"):
                val = getattr(s, attr, None)
                if val and isinstance(val, str):
                    return val
            if getattr(s, "primary_artist", None):
                return getattr(s.primary_artist, "name", None) or artist_name
            return artist_name

        # Cover art: Deezer → iTunes → Genius attributes
        cover_art = fetch_deezer_cover(song_title, artist_name)
        if not cover_art:
            for attr in ("song_art_image_url", "song_art_image_thumbnail_url",
                         "header_image_url", "header_image_thumbnail_url"):
                url = getattr(song, attr, None)
                if url and isinstance(url, str) and url.startswith("http"):
                    cover_art = url
                    break

        print(f"[Ingest] Cover art: {cover_art or 'not found'}")

        def _album_name(s):
            a = getattr(s, "album", None)
            if not a: return None
            if isinstance(a, dict): return a.get("name")
            return getattr(a, "name", None)

        meta = {
            "title":        getattr(song, "title", song_title),
            "artist":       _artist_str(song),
            "album":        _album_name(song),
            "release_date": getattr(song, "release_date_for_display", None) or None,
            "cover_art":    cover_art,
            "artist_image": (
                getattr(song.primary_artist, "image_url", None)
                if getattr(song, "primary_artist", None) else None
            ),
            "genius_url":   getattr(song, "url", None) or None,
        }
        lyrics = getattr(song, "lyrics", None)
        return meta, lyrics

    except Exception as e:
        print(f"[Genius Error] {e}")
        return {"title": song_title, "artist": artist_name}, None


def fetch_song_metadata(song_title: str, artist_name: str) -> dict:
    """
    Fetch rich metadata for a song from the Genius API.

    Returns a dict with these keys (all optional — may be None if unavailable):
        title        : canonical song title from Genius
        artist       : artist name(s) string
        album        : album name, or None
        release_date : human-readable date string, e.g. "March 29, 2019"
        cover_art    : URL to album/song art image
        artist_image : URL to the primary artist's profile image
        genius_url   : link to the Genius song page

    Returns an empty dict if the song cannot be found.
    """
    genius = _get_genius_client()
    try:
        song = genius.search_song(song_title, artist_name)
        if not song:
            return {}

        # artist_names is not always present — fall back through multiple attributes
        def _artist_str(s):
            for attr in ("artist_names", "artist"):
                val = getattr(s, attr, None)
                if val and isinstance(val, str):
                    return val
            if getattr(s, "primary_artist", None):
                return getattr(s.primary_artist, "name", None) or artist_name
            return artist_name

        def _alb(s):
            a = getattr(s, "album", None)
            if not a: return None
            return a.get("name") if isinstance(a, dict) else getattr(a, "name", None)

        meta = {
            "title":        getattr(song, "title", song_title),
            "artist":       _artist_str(song),
            "album":        _alb(song),
            "release_date": getattr(song, "release_date_for_display", None) or None,
            "cover_art":    getattr(song, "song_art_image_url", None) or None,
            "artist_image": (
                getattr(song.primary_artist, "image_url", None)
                if getattr(song, "primary_artist", None) else None
            ),
            "genius_url":   getattr(song, "url", None) or None,
        }
        return meta

    except Exception as e:
        print(f"[Genius Metadata Error] {e}")
        # Return a minimal fallback so the UI still shows something
        return {"title": song_title, "artist": artist_name}


def fetch_wikipedia_article(search_term: str) -> str | None:
    """
    Fetch the summary + full text of a Wikipedia article.
    search_term can be an artist name, album name, or song name.

    Returns:
        The article text, or None if the page doesn't exist.
    """
    wiki = wikipediaapi.Wikipedia(
        language="en",
        user_agent=WIKI_USER_AGENT
    )
    page = wiki.page(search_term)
    if page.exists():
        return page.text   # full article text (may be very long)
    return None


# ── Step 2: Chunk ─────────────────────────────────────────────────────────────

def chunk_lyrics(lyrics: str, lines_per_chunk: int = LYRIC_CHUNK_LINES) -> list[str]:
    """
    Split lyrics into overlapping groups of lines.

    Why line-based chunking for lyrics?
      Lyrics are short lines. A paragraph-based split would yield either
      gigantic chunks (entire verses mashed together) or single-line chunks
      that are too small for semantic meaning. Groups of ~8 lines capture
      a verse + hook naturally.

    Args:
        lyrics         : raw lyrics string
        lines_per_chunk: number of lines per chunk (default 8)

    Returns:
        List of chunk strings.
    """
    # Clean up the raw Genius text — remove contributor count line at top
    lyrics = re.sub(r"^\d+\s*Contributors?.*\n", "", lyrics, flags=re.IGNORECASE)
    # Remove "Embed" at the end (Genius artifact)
    lyrics = re.sub(r"\d*Embed$", "", lyrics).strip()

    lines = [line.strip() for line in lyrics.split("\n") if line.strip()]

    chunks = []
    for i in range(0, len(lines), lines_per_chunk):
        chunk = "\n".join(lines[i: i + lines_per_chunk])
        if chunk:
            chunks.append(chunk)

    return chunks


def chunk_article(article_text: str, max_chars: int = 1500) -> list[str]:
    """
    Split a Wikipedia article into paragraph-sized chunks.

    Why paragraph-based for articles?
      Wikipedia paragraphs are semantically coherent units. Splitting by
      paragraph preserves context (e.g. "Early life", "Discography" sections
      don't bleed into each other).

    Long paragraphs are further split at max_chars to avoid token limit issues.

    Args:
        article_text : full Wikipedia article text
        max_chars    : maximum character length per chunk

    Returns:
        List of chunk strings.
    """
    paragraphs = [p.strip() for p in article_text.split("\n\n") if p.strip()]

    chunks = []
    for para in paragraphs:
        if len(para) <= max_chars:
            chunks.append(para)
        else:
            # Split long paragraphs at sentence boundaries
            sentences = re.split(r"(?<=[.!?])\s+", para)
            current = ""
            for sentence in sentences:
                if len(current) + len(sentence) <= max_chars:
                    current += (" " if current else "") + sentence
                else:
                    if current:
                        chunks.append(current)
                    current = sentence
            if current:
                chunks.append(current)

    return chunks


# ── Step 3 + 4: Embed & Store ─────────────────────────────────────────────────

def store_chunks(
    chunks: list[str],
    source_label: str,
    collection: chromadb.Collection
) -> None:
    """
    Embed each chunk using sentence-transformers (via ChromaDB's embedding
    function) and store them in the ChromaDB collection.

    ChromaDB requires each document to have a unique ID.
    We build IDs like: "Genius-Blinding Lights-0", "Genius-Blinding Lights-1"

    Args:
        chunks       : list of text chunks to store
        source_label : human-readable source tag, e.g. "Genius - Blinding Lights"
        collection   : the ChromaDB collection to write into
    """
    if not chunks:
        return

    ids       = [f"{source_label}-{i}" for i in range(len(chunks))]
    metadatas = [{"source": source_label} for _ in chunks]

    # upsert = insert or overwrite — safe even if IDs already exist
    collection.upsert(
        documents=chunks,
        ids=ids,
        metadatas=metadatas
    )
    print(f"[ChromaDB] Stored {len(chunks)} chunks from '{source_label}'")


# ── Main Public Function ───────────────────────────────────────────────────────

def ingest(
    song_title: str,
    artist_name: str
) -> tuple[list[str], list[str], chromadb.Collection, dict]:
    """
    Full pipeline: fetch → chunk → embed → store.
    Also fetches rich song metadata (cover art, artist image, etc.) for the UI.

    Returns:
        lyric_chunks   : list of lyric text chunks  (for BM25 indexing)
        article_chunks : list of article text chunks (for BM25 indexing)
        collection     : the live ChromaDB collection (for vector search)
        song_meta      : dict with cover_art, artist_image, album, release_date, etc.
    """
    # Always start fresh — wipes the old DB directory so there are no
    # stale chunks from previous songs and no version-mismatch errors
    collection     = _get_chroma_collection(fresh=True)
    lyric_chunks   = []
    article_chunks = []

    # ── Metadata + Lyrics (single Genius API call) ────────────────────────
    print(f"[Ingest] Fetching metadata + lyrics: '{song_title}' by {artist_name}...")
    song_meta, lyrics = fetch_song_and_lyrics(song_title, artist_name)

    if lyrics:
        lyric_chunks = chunk_lyrics(lyrics)
        source_label = f"Genius - {song_title} by {artist_name}"
        store_chunks(lyric_chunks, source_label, collection)
    else:
        print(f"[Ingest] Lyrics not found for '{song_title}'")

    # ── Wikipedia — Artist ────────────────────────────────────────────────
    print(f"[Ingest] Fetching Wikipedia article for: {artist_name}...")
    artist_article = fetch_wikipedia_article(artist_name)
    if artist_article:
        artist_chunks = chunk_article(artist_article)
        store_chunks(artist_chunks, f"Wikipedia - {artist_name}", collection)
        article_chunks.extend(artist_chunks)
    else:
        print(f"[Ingest] Wikipedia article not found for '{artist_name}'")

    # ── Wikipedia — Song/Album ────────────────────────────────────────────
    print(f"[Ingest] Fetching Wikipedia article for: {song_title}...")
    song_article = fetch_wikipedia_article(f"{song_title} ({artist_name} song)")
    if not song_article:
        song_article = fetch_wikipedia_article(song_title)
    if song_article:
        song_chunks = chunk_article(song_article)
        store_chunks(song_chunks, f"Wikipedia - {song_title}", collection)
        article_chunks.extend(song_chunks)
    else:
        print(f"[Ingest] Wikipedia article not found for '{song_title}'")

    return lyric_chunks, article_chunks, collection, song_meta
