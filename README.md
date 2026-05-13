---
title: LyricLens
emoji: 🎵
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# LyricLens

An emotionally intelligent music research assistant built on Hybrid RAG with agentic query routing. Ask anything about a song — emotional atmosphere, lyrical themes, artist background — and get grounded, cited answers.

## How It Works

```
Query → Router → Hybrid Retriever (BM25 + Vector) → LLM Agent → Response
```

Every query is classified by a two-stage router (regex fast-path → Groq LLM fallback) that tunes retrieval weights and source priority per intent. Factual queries get keyword-heavy BM25 against Wikipedia. Emotional queries get semantic vector search against lyrics. Both score lists are min-max normalised then merged with router-supplied weights.

## Stack

| Layer | Technology |
|---|---|
| LLM | Groq `llama-3.3-70b-versatile` |
| Vector DB | ChromaDB + `all-MiniLM-L6-v2` |
| Keyword Search | BM25Okapi (rank-bm25) |
| Data Sources | Genius API · Wikipedia |
| Backend | FastAPI + Uvicorn |
| Frontend | React 18 — single HTML file, no build step |

## Structure

```
lyriclens/
├── api.py          # FastAPI server, query expansion, session state
├── router.py       # Query router — regex + LLM classification
├── retriever.py    # Hybrid BM25 + vector search, weighted merge
├── ingest.py       # Fetch → chunk → embed → store pipeline
├── agent.py        # Prompt engineering, 3 modes, recommendations
├── memory.py       # Sliding-window conversation memory (6 turns)
└── frontend/
    └── index.html  # React UI
```

## Setup

Requires Python 3.10+, a [Genius API token](https://genius.com/api-clients), and a [Groq API key](https://console.groq.com).

```bash
pip install -r requirements.txt
```

Create `.env`:
```
GENIUS_ACCESS_TOKEN=your_token
GROQ_API_KEY=your_key
```

```bash
uvicorn api:app --reload --port 8000
```

Open `http://localhost:8000`

## API

| Method | Endpoint | Body |
|--------|----------|------|
| POST | `/api/ingest` | `{ song_title, artist_name }` |
| POST | `/api/query` | `{ query, mode }` — `quick` · `deep` · `vibe` |
| POST | `/api/clear` | — |
| POST | `/api/reset` | — |
| GET | `/api/status` | — |
