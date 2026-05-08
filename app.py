# ─────────────────────────────────────────────────────────────────────────────
# app.py — LyricLens · Dark Theme · Two-column layout (no manual panel divs)
# ─────────────────────────────────────────────────────────────────────────────

import re
import streamlit as st
from memory    import ConversationMemory
from ingest    import ingest
from retriever import hybrid_retrieve
from agent     import generate_answer, format_sources, suggest_related_songs

st.set_page_config(
    page_title="LyricLens",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
}
#MainMenu, footer, [data-testid="stSidebar"],
[data-testid="collapsedControl"], header { display: none !important; }

/* ── Page background ── */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(160deg, #080415 0%, #100825 55%, #090d25 100%);
    min-height: 100vh;
}
[data-testid="stMain"] { background: transparent; }
.block-container { padding: 1.4rem 1.8rem 1.5rem !important; max-width: 1280px; }

/* ── Style BOTH columns as dark panels via CSS (no manual HTML divs needed) ── */
[data-testid="stColumn"] > div:first-child {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(124,58,237,0.2);
    border-radius: 18px;
    padding: 1.3rem 1.2rem !important;
    min-height: 78vh;
}

/* ── App header ── */
.ll-header {
    background: linear-gradient(135deg, #5b21b6 0%, #4338ca 50%, #7c3aed 100%);
    border-radius: 16px;
    padding: 1.2rem 1.8rem;
    margin-bottom: 1.4rem;
    color: white;
    box-shadow: 0 8px 40px rgba(124,58,237,0.35);
    display: flex;
    align-items: center;
    gap: 1rem;
    border: 1px solid rgba(167,139,250,0.2);
}
.ll-header h1 { margin: 0; font-size: 1.7rem; font-weight: 800; letter-spacing: -0.5px; }
.ll-header p  { margin: 0; opacity: 0.8; font-size: 0.85rem; margin-top: 2px; }

/* ── Section labels ── */
.sec-lbl {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: #a78bfa;
    margin: 14px 0 6px;
    display: block;
}

/* ── Inputs ── */
.stTextInput input {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(124,58,237,0.35) !important;
    border-radius: 10px !important;
    color: #e2d9f3 !important;
    font-size: 0.87rem !important;
}
.stTextInput input::placeholder { color: #6b5f84 !important; }
.stTextInput input:focus {
    border-color: #7c3aed !important;
    background: rgba(124,58,237,0.1) !important;
    box-shadow: 0 0 0 3px rgba(124,58,237,0.15) !important;
}
.stTextInput label { color: #9d8abf !important; font-size: 0.78rem !important; }

/* ── Radio ── */
[data-testid="stRadio"] label { color: #c4b5fd !important; font-size: 0.84rem !important; }
[data-testid="stRadio"] > div { gap: 3px !important; }

/* ── Buttons ── */
.stButton > button {
    background: rgba(255,255,255,0.05) !important;
    color: #c4b5fd !important;
    border: 1px solid rgba(124,58,237,0.35) !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.83rem !important;
    transition: all 0.18s !important;
}
.stButton > button:hover {
    background: rgba(124,58,237,0.2) !important;
    border-color: #7c3aed !important;
    transform: translateY(-1px) !important;
}
[data-testid="stBaseButton-primary"] button {
    background: linear-gradient(135deg, #7c3aed, #6d28d9) !important;
    color: white !important;
    border: none !important;
}

/* ── Album card ── */
.album-card {
    background: linear-gradient(135deg, rgba(124,58,237,0.15), rgba(67,56,202,0.1));
    border: 1px solid rgba(124,58,237,0.3);
    border-radius: 14px;
    padding: 12px 14px;
    margin-top: 8px;
}
.album-title  { font-size: 0.95rem; font-weight: 700; color: #f0ebff; }
.album-artist { font-size: 0.8rem; color: #a78bfa; font-weight: 600; margin-top: 2px; }
.album-meta   { font-size: 0.72rem; color: #6b5f84; margin-top: 5px; line-height: 1.7; }

/* ── Example queries ── */
.example-q {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(124,58,237,0.15);
    border-radius: 9px;
    padding: 5px 10px;
    font-size: 0.77rem;
    color: #9d8abf;
    margin: 3px 0;
    display: block;
}

/* ── Status badges ── */
.badge-ok   { background: rgba(16,185,129,0.15); color: #6ee7b7; border: 1px solid rgba(16,185,129,0.3);  border-radius:20px; padding:3px 10px; font-size:0.73rem; font-weight:600; display:inline-block; }
.badge-warn { background: rgba(245,158,11,0.12);  color: #fcd34d; border: 1px solid rgba(245,158,11,0.25); border-radius:20px; padding:3px 10px; font-size:0.73rem; font-weight:600; display:inline-block; }

/* ── Chat bubbles ── */
.bubble-user-wrap { display:flex; flex-direction:column; align-items:flex-end;  margin:8px 0; }
.bubble-ai-wrap   { display:flex; flex-direction:column; align-items:flex-start; margin:8px 0; }
.blbl { font-size:0.65rem; font-weight:700; letter-spacing:0.8px; text-transform:uppercase; opacity:0.45; margin-bottom:4px; }

.bubble-user {
    background: linear-gradient(135deg, #7c3aed, #6d28d9);
    color: white;
    padding: 10px 14px;
    border-radius: 18px 18px 4px 18px;
    max-width: 80%;
    font-size: 0.9rem;
    line-height: 1.55;
    box-shadow: 0 4px 16px rgba(124,58,237,0.3);
}
.bubble-ai {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(124,58,237,0.22);
    border-radius: 18px 18px 18px 4px;
    padding: 14px 16px;
    max-width: 96%;
    font-size: 0.88rem;
    line-height: 1.65;
    color: #e2d9f3;
    box-shadow: 0 4px 20px rgba(0,0,0,0.2);
}

/* Structured sections inside bubble */
.bubble-ai .sec-header {
    display: block;
    font-size: 0.8rem;
    font-weight: 700;
    color: #a78bfa;
    margin: 14px 0 6px;
    padding-bottom: 5px;
    border-bottom: 1px solid rgba(167,139,250,0.18);
}
.bubble-ai .sec-header:first-child { margin-top: 0; }
.bubble-ai ul { margin: 4px 0 10px; padding-left: 16px; }
.bubble-ai li { color: #c4b5fd; margin: 5px 0; font-size: 0.87rem; line-height: 1.5; }
.bubble-ai p  { margin: 4px 0 8px; color: #e2d9f3; }
.bubble-ai strong { color: #f0ebff; }
.bubble-ai .cite  { color: #7c3aed; font-size: 0.8em; }
.bubble-ai .sources-line {
    margin-top: 12px;
    padding-top: 8px;
    border-top: 1px solid rgba(167,139,250,0.12);
    font-size: 0.76rem;
    color: #6b5f84;
}

/* ── Source pills ── */
.src-pill {
    display: inline-block;
    background: rgba(124,58,237,0.15);
    color: #a78bfa;
    border: 1px solid rgba(124,58,237,0.25);
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.72rem;
    font-weight: 600;
    margin: 2px;
}

/* ── Related songs ── */
.related-hdr {
    font-size: 0.68rem; font-weight: 700; letter-spacing: 0.8px;
    text-transform: uppercase; color: #7c3aed; margin: 14px 0 8px;
    display: block;
}
.rel-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(124,58,237,0.2);
    border-radius: 12px;
    padding: 11px 14px;
    min-height: 85px;
    transition: border-color 0.15s, background 0.15s;
}
.rel-card:hover { border-color: #7c3aed; background: rgba(124,58,237,0.08); }
.rel-title  { font-size: 0.85rem; font-weight: 700; color: #f0ebff; }
.rel-artist { font-size: 0.75rem; color: #a78bfa; font-weight: 600; margin-top: 2px; }
.rel-reason { font-size: 0.73rem; color: #6b5f84; margin-top: 4px; line-height: 1.4; }

/* ── Empty state ── */
.empty-state { text-align:center; padding:4rem 1rem; }
.empty-icon  { font-size: 3rem; margin-bottom: 0.6rem; }
.empty-title { font-size: 1rem; font-weight: 600; color: #6b5f84; margin-bottom: 0.3rem; }
.empty-sub   { font-size: 0.82rem; color: #4a4060; }

/* ── Expander ── */
[data-testid="stExpander"] {
    border: 1px solid rgba(124,58,237,0.2) !important;
    border-radius: 10px !important;
    background: rgba(255,255,255,0.02) !important;
}
[data-testid="stExpander"] summary { color: #9d8abf !important; font-size: 0.8rem !important; }

/* ── Chat input ── */
[data-testid="stChatInput"] textarea {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(124,58,237,0.3) !important;
    border-radius: 14px !important;
    color: #e2d9f3 !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: #4a4060 !important; }

hr { border-color: rgba(124,58,237,0.15) !important; margin: 11px 0 !important; }
</style>
""", unsafe_allow_html=True)


# ── Markdown → HTML converter ─────────────────────────────────────────────────
def _md_to_html(text: str) -> str:
    lines  = text.split("\n")
    html   = []
    in_ul  = False

    for line in lines:
        s = line.strip()
        if s.startswith("## "):
            if in_ul: html.append("</ul>"); in_ul = False
            html.append(f'<span class="sec-header">{s[3:]}</span>')
        elif s.startswith("- ") or s.startswith("* "):
            if not in_ul: html.append("<ul>"); in_ul = True
            html.append(f"<li>{_inline(s[2:])}</li>")
        elif s.lower().startswith("sources:"):
            if in_ul: html.append("</ul>"); in_ul = False
            html.append(f'<div class="sources-line">📎 {_inline(s[8:].strip())}</div>')
        elif s == "":
            if in_ul: html.append("</ul>"); in_ul = False
        else:
            if in_ul: html.append("</ul>"); in_ul = False
            html.append(f"<p>{_inline(s)}</p>")

    if in_ul: html.append("</ul>")
    return "\n".join(html)

def _inline(t: str) -> str:
    t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)
    t = re.sub(r"\[([^\]]+)\]",  r'<span class="cite">[\1]</span>', t)
    return t


# ── Session State ─────────────────────────────────────────────────────────────
for k, v in {
    "memory":       ConversationMemory(window_size=6),
    "chat_history": [],
    "all_chunks":   [],
    "collection":   None,
    "ingested_for": None,
    "song_meta":    {},
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="ll-header">
    <span style="font-size:2rem">🎵</span>
    <div>
        <h1>LyricLens</h1>
        <p>AI-powered music research · lyrics · themes · artists · emotional vibes</p>
    </div>
</div>
""", unsafe_allow_html=True)

left, right = st.columns([1.1, 2.9], gap="large")


# ══════════════════════════════════════════════════════════════════════════════
# LEFT — Controls (no manual panel div — styled via CSS on the column itself)
# ══════════════════════════════════════════════════════════════════════════════
with left:
    st.markdown('<span class="sec-lbl">① Song to explore</span>', unsafe_allow_html=True)
    song_title  = st.text_input("Song",   placeholder="e.g. FEIN",          label_visibility="collapsed")
    artist_name = st.text_input("Artist", placeholder="e.g. Travis Scott",   label_visibility="collapsed")

    st.markdown('<span class="sec-lbl">② Mode</span>', unsafe_allow_html=True)
    mode = st.radio(
        "mode", ["quick", "deep"],
        format_func=lambda x: "⚡ Quick — punchy answer" if x == "quick" else "🔬 Deep — full analysis",
        label_visibility="collapsed"
    )
    top_k = 3 if mode == "quick" else 8

    st.markdown('<span class="sec-lbl">③ Actions</span>', unsafe_allow_html=True)
    b1, b2    = st.columns(2)
    load_btn  = b1.button("🚀 Load",  use_container_width=True, type="primary")
    clear_btn = b2.button("🗑️ Clear", use_container_width=True)

    if clear_btn:
        st.session_state.memory       = ConversationMemory(window_size=6)
        st.session_state.chat_history = []
        st.rerun()

    if load_btn:
        if not song_title or not artist_name:
            st.error("Enter a song title and artist name.")
        else:
            key = f"{song_title.lower()}|{artist_name.lower()}"
            if st.session_state.ingested_for == key:
                st.success("✅ Already loaded!")
            else:
                with st.spinner("Fetching data..."):
                    lc, ac, col, meta = ingest(song_title, artist_name)
                    chunks = lc + ac
                    if not chunks:
                        st.error("No data found. Check API keys or try another song.")
                    else:
                        st.session_state.all_chunks    = chunks
                        st.session_state.collection    = col
                        st.session_state.ingested_for  = key
                        st.session_state.song_meta     = meta
                        st.session_state.memory        = ConversationMemory(window_size=6)
                        st.session_state.chat_history  = []
                        st.success(f"✅ {len(chunks)} chunks loaded!")

    if st.session_state.collection:
        st.markdown('<span class="badge-ok">📀 Song loaded</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge-warn">⚠️ Nothing loaded</span>', unsafe_allow_html=True)

    st.divider()

    # Album / Artist Card
    meta = st.session_state.song_meta
    if meta:
        st.markdown('<span class="sec-lbl">🎧 Now Exploring</span>', unsafe_allow_html=True)
        i1, i2 = st.columns(2)
        with i1:
            if meta.get("cover_art"):
                st.image(meta["cover_art"], use_container_width=True)
        with i2:
            if meta.get("artist_image"):
                st.image(meta["artist_image"], use_container_width=True)
        album_ln = f"💿 {meta['album']}<br>" if meta.get("album")        else ""
        date_ln  = f"📅 {meta['release_date']}" if meta.get("release_date") else ""
        st.markdown(f"""
        <div class="album-card">
            <div class="album-title">{meta.get('title','')}</div>
            <div class="album-artist">🎤 {meta.get('artist','')}</div>
            <div class="album-meta">{album_ln}{date_ln}</div>
        </div>""", unsafe_allow_html=True)
        st.divider()

    st.markdown('<span class="sec-lbl">💡 Try asking</span>', unsafe_allow_html=True)
    for ex in [
        "Explain the story behind this song",
        "What are the core emotional themes?",
        "What's the musical atmosphere like?",
        "How does this fit the artist's evolution?",
        "Why does this song matter culturally?",
    ]:
        st.markdown(f'<span class="example-q">› {ex}</span>', unsafe_allow_html=True)

    st.divider()
    st.caption("Powered by Groq · ChromaDB · Genius API · Wikipedia")


# ══════════════════════════════════════════════════════════════════════════════
# RIGHT — Chat (no manual panel div — styled via CSS on the column itself)
# ══════════════════════════════════════════════════════════════════════════════
with right:
    if not st.session_state.chat_history:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">🎧</div>
            <div class="empty-title">Load a song on the left to begin</div>
            <div class="empty-sub">Ask about lyrics, themes, emotional vibes, artist history…</div>
        </div>""", unsafe_allow_html=True)
    else:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f"""
                <div class="bubble-user-wrap">
                    <div class="blbl" style="color:#a78bfa;">You</div>
                    <div class="bubble-user">{msg["content"]}</div>
                </div>""", unsafe_allow_html=True)

            elif msg["role"] == "assistant":
                html_content = _md_to_html(msg["content"])
                st.markdown(f"""
                <div class="bubble-ai-wrap">
                    <div class="blbl" style="color:#6b5f84;">LyricLens</div>
                    <div class="bubble-ai">{html_content}</div>
                </div>""", unsafe_allow_html=True)

                if msg.get("source_list"):
                    with st.expander("📚 Sources", expanded=False):
                        for src in msg["source_list"]:
                            st.markdown(f'<span class="src-pill">📄 {src}</span>',
                                        unsafe_allow_html=True)

                if msg.get("related_songs"):
                    st.markdown('<span class="related-hdr">🎵 Related Songs You Might Love</span>',
                                unsafe_allow_html=True)
                    rcols = st.columns(min(len(msg["related_songs"]), 3))
                    for i, s in enumerate(msg["related_songs"]):
                        with rcols[i % 3]:
                            st.markdown(f"""
                            <div class="rel-card">
                                <div class="rel-title">{s.get('title','')}</div>
                                <div class="rel-artist">{s.get('artist','')}</div>
                                <div class="rel-reason">{s.get('reason','')}</div>
                            </div>""", unsafe_allow_html=True)

    user_query = st.chat_input(
        "Ask about lyrics, themes, artist history, emotional vibes...",
        disabled=(st.session_state.collection is None)
    )


# ── Handle query ──────────────────────────────────────────────────────────────
if user_query:
    if not st.session_state.collection or not st.session_state.all_chunks:
        st.warning("⚠️ Load a song first!")
        st.stop()

    with st.spinner("🔍 Retrieving sources..."):
        retrieved = hybrid_retrieve(
            query      = user_query,
            all_chunks = st.session_state.all_chunks,
            collection = st.session_state.collection,
            top_k      = top_k
        )
    with st.spinner("🤖 Writing analysis..."):
        mem_text        = st.session_state.memory.format()
        answer, sources = generate_answer(
            query            = user_query,
            retrieved_chunks = retrieved,
            memory_text      = mem_text,
            mode             = mode
        )
    with st.spinner("🎵 Finding related songs..."):
        m       = st.session_state.song_meta
        st_name = m.get("title",  song_title  or "")
        ar_name = m.get("artist", artist_name or "")
        related = suggest_related_songs(st_name, ar_name, retrieved)

    st.session_state.memory.add("user",      user_query)
    st.session_state.memory.add("assistant", answer)
    st.session_state.chat_history.append({
        "role": "user", "content": user_query,
        "source_list": [], "related_songs": []
    })
    st.session_state.chat_history.append({
        "role": "assistant", "content": answer,
        "source_list": sources, "related_songs": related
    })
    st.rerun()
