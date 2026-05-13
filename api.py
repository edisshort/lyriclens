# ─────────────────────────────────────────────────────────────────────────────
# api.py — FastAPI backend for LyricLens
# Run with:  uvicorn api:app --reload --port 8000
# ─────────────────────────────────────────────────────────────────────────────

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
import os
import traceback
import urllib.request

from memory    import ConversationMemory
from ingest    import ingest
from retriever import hybrid_retrieve
from agent     import generate_answer, format_sources, suggest_related_songs
from router    import route_query

app = FastAPI(title="LyricLens API")

# Allow requests from React dev server (localhost:5173) and same origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory session state (single-user app) ─────────────────────────────────
_state = {
    "collection":   None,
    "all_chunks":   [],
    "song_meta":    {},
    "ingested_for": None,
    "memory":       ConversationMemory(window_size=6),
}


# ── Request / Response models ─────────────────────────────────────────────────

class IngestRequest(BaseModel):
    song_title:  str
    artist_name: str

class QueryRequest(BaseModel):
    query: str
    mode:  str = "quick"   # "quick" | "deep"

class ClearRequest(BaseModel):
    pass


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/api/status")
def status():
    """Return current load state."""
    return {
        "loaded":    _state["collection"] is not None,
        "song_meta": _state["song_meta"],
        "chunks":    len(_state["all_chunks"]),
    }


@app.post("/api/ingest")
def api_ingest(req: IngestRequest):
    """Fetch lyrics + Wikipedia, embed, store in ChromaDB."""
    key = f"{req.song_title.lower()}|{req.artist_name.lower()}"
    if _state["ingested_for"] == key:
        return {
            "status":       "already_loaded",
            "chunks_count": len(_state["all_chunks"]),
            "song_meta":    _state["song_meta"],
        }

    try:
        lc, ac, col, meta = ingest(req.song_title, req.artist_name)
    except Exception as e:
        traceback.print_exc()   # prints full stack trace to the terminal
        raise HTTPException(status_code=500, detail=str(e))

    chunks = lc + ac
    if not chunks:
        raise HTTPException(status_code=404, detail="No data found. Check song title / API keys.")

    _state["collection"]   = col
    _state["all_chunks"]   = chunks
    _state["song_meta"]    = meta
    _state["ingested_for"] = key
    _state["memory"]       = ConversationMemory(window_size=6)

    return {
        "status":       "ok",
        "chunks_count": len(chunks),
        "song_meta":    meta,
    }


def _expand_query(query: str, song_meta: dict) -> str:
    """
    Auto-inject current song/artist context into the query so that
    vague questions like "what's it about?" or "how does this feel?"
    always retrieve the right chunks.
    """
    title  = song_meta.get("title",  "")
    artist = song_meta.get("artist", "")
    album  = song_meta.get("album",  "")

    if not title and not artist:
        return query

    # Only prepend context if query doesn't already name the song/artist
    q_lower = query.lower()
    already_specific = (
        title.lower()  in q_lower or
        artist.lower() in q_lower
    )
    if already_specific:
        return query

    context = f'[Song: "{title}" by {artist}'
    if album:
        context += f', Album: {album}'
    context += '] '
    return context + query


@app.post("/api/query")
def api_query(req: QueryRequest):
    """Retrieve → Generate → Related songs."""
    if not _state["collection"] or not _state["all_chunks"]:
        raise HTTPException(status_code=400, detail="No song loaded. Call /api/ingest first.")

    # ── Step 1: Route the query ───────────────────────────────────────────
    strategy = route_query(req.query)
    # In vibe/deep mode, allow more chunks than the router's default
    top_k = strategy["top_k"]
    if req.mode == "deep" and top_k < 7:
        top_k = 8
    elif req.mode == "quick" and top_k > 5:
        top_k = 4

    # ── Step 2: Expand query with song context ────────────────────────────
    expanded_query = _expand_query(req.query, _state["song_meta"])

    # ── Step 3: Retrieve with routing-tuned parameters ────────────────────
    retrieved = hybrid_retrieve(
        query           = expanded_query,
        all_chunks      = _state["all_chunks"],
        collection      = _state["collection"],
        top_k           = top_k,
        bm25_weight     = strategy["bm25_weight"],
        vector_weight   = strategy["vector_weight"],
        source_priority = strategy["source_priority"],
    )

    # Generate
    mem_text        = _state["memory"].format()
    answer, sources = generate_answer(
        query            = req.query,
        retrieved_chunks = retrieved,
        memory_text      = mem_text,
        mode             = req.mode,
    )

    # Related songs
    meta    = _state["song_meta"]
    related = suggest_related_songs(
        song_title   = meta.get("title",  ""),
        artist_name  = meta.get("artist", ""),
        retrieved_chunks = retrieved,
    )

    # Update memory
    _state["memory"].add("user",      req.query)
    _state["memory"].add("assistant", answer)

    return {
        "answer":        answer,
        "sources":       sources,
        "related_songs": related,
        "route":         strategy["query_type"],        # e.g. "emotional"
        "route_desc":    strategy["description"],       # human-readable
    }


@app.post("/api/clear")
def api_clear():
    """Reset conversation memory."""
    _state["memory"] = ConversationMemory(window_size=6)
    return {"status": "cleared"}


@app.post("/api/reset")
def api_reset():
    """Full reset — clear everything including loaded song."""
    _state["collection"]   = None
    _state["all_chunks"]   = []
    _state["song_meta"]    = {}
    _state["ingested_for"] = None
    _state["memory"]       = ConversationMemory(window_size=6)
    return {"status": "reset"}


# ── Image proxy — avoids browser CORS/hotlink blocks on Genius image URLs ────
@app.get("/api/image")
def proxy_image(url: str):
    """Fetch a remote image server-side and return it to the browser."""
    if not url.startswith("http"):
        raise HTTPException(status_code=400, detail="Invalid URL")
    # Try multiple User-Agent strings in case one is blocked
    for ua in [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36",
        "iTunes/12.0 (Macintosh)",
        "curl/7.68.0",
    ]:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": ua})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data         = resp.read()
                content_type = resp.headers.get_content_type() or "image/jpeg"
            return Response(content=data, media_type=content_type)
        except Exception:
            continue
    raise HTTPException(status_code=502, detail="Image fetch failed after retries")


# ── Serve React frontend ───────────────────────────────────────────────────────
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "frontend")

if os.path.isdir(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/")
    def serve_index():
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))
