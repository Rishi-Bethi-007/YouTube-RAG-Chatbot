import streamlit as st
from src.init_db import init_db
from src.memory import update_summary
from src.config import NAMESPACE, CHUNK_CHARS, CHUNK_OVERLAP_CHARS
from src.youtube_ids import extract_video_ids
from src.transcripts import load_or_fetch_transcript
from src.chunking import chunk_transcript
from src.ingest import ingest_video
from src.retrieve import answer_question

init_db()

st.set_page_config(page_title="YouTube RAG (Prod-style)", layout="wide")
st.title("YouTube Chatbot")

with st.sidebar:
    st.subheader("Dataset")
    namespace = st.text_input("Namespace", value=NAMESPACE)
    force = st.checkbox("Force re-ingest videos", value=False)

    st.subheader("Chunking")
    chunk_chars = st.slider("Chunk chars", 400, 1800, CHUNK_CHARS, 50)
    overlap_chars = st.slider("Overlap chars", 0, 500, CHUNK_OVERLAP_CHARS, 10)

st.write("Paste a video URL, multiple URLs (one per line), or a playlist URL.")
input_text = st.text_area("Input", height=120)

c1, c2 = st.columns(2)

if c1.button("Build / Update Index"):
    if not input_text.strip():
        st.error("Paste a URL or playlist.")
        st.stop()

    video_ids = extract_video_ids(input_text)
    st.info(f"Found {len(video_ids)} videos.")

    total_new = 0
    skipped = 0

    for vid in video_ids:
        with st.spinner(f"Ingesting {vid} ..."):
            try:
                items = load_or_fetch_transcript(vid)   # may raise
            except Exception as e:
                st.warning(f"Skipping {vid}: {e}")
                continue

            chunks = chunk_transcript(items, chunk_chars=chunk_chars, overlap_chars=overlap_chars)
            stats = ingest_video(video_id=vid, title=f"YouTube {vid}", chunks=chunks, namespace=namespace, force=force)
            if stats.get("skipped"):
                skipped += 1
            total_new += int(stats.get("new_chunks", 0))

    st.success(f"Done. New chunks: {total_new}. Skipped videos: {skipped}.")


st.divider()
st.subheader("Chat")

if "messages" not in st.session_state:
    st.session_state.messages = []  # [{"role":"user"|"assistant", "content": "..."}]
if "summary" not in st.session_state:
    st.session_state.summary = ""

# Render chat history
for m in st.session_state.messages:
    st.chat_message(m["role"]).write(m["content"])

user_text = st.chat_input("Ask about the videos...")

if user_text:
    # append user message
    st.session_state.messages.append({"role": "user", "content": user_text})
    st.chat_message("user").write(user_text)

    # recent turns for retrieval (last 6 messages)
    recent_turns = st.session_state.messages[-6:]

    with st.spinner("Retrieving + reranking + generating..."):
        out = answer_question(
            user_text,
            namespace=namespace,
            summary=st.session_state.summary,
            recent_turns=recent_turns
        )

    # append assistant message
    st.session_state.messages.append({"role": "assistant", "content": out["answer"]})
    st.chat_message("assistant").write(out["answer"])

    with st.expander("Sources"):
        for s in out["sources"]:
            st.markdown(f"- [{s['title']} | {s['video_id']} @ {s['start']}s]({s['url']})")

    with st.expander("Latency + Cache"):
        st.write(out["timings"])
        st.write(out["cache"])

    # Update summary using only the last user+assistant turn (fast + stable)
    try:
        st.session_state.summary = update_summary(
            st.session_state.summary,
            new_messages=[
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": out["answer"]},
            ],
            max_chars=1500
        )
    except Exception:
        # If summarization fails, do not break chat
        pass

with st.sidebar:
    st.subheader("Memory")
    st.caption("This summary is used to keep chat context without sending the full history every turn.")
    st.text_area("Conversation summary", st.session_state.summary, height=150)

    if st.button("Reset chat"):
        st.session_state.messages = []
        st.session_state.summary = ""
        st.rerun()
