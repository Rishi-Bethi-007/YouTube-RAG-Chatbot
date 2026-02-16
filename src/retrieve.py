import json
import time
from typing import Any
from sqlalchemy import select
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from src.config import OPENAI_API_KEY, EMBED_MODEL, CHAT_MODEL, FETCH_K, TOP_K, RERANK_TOP_N
from src.cache import get_json, set_json, sha1
from src.db import SessionLocal
from src.models import Chunk, Video
from src.pinecone_store import ensure_index
from src.rewrite import rewrite_query
from src.rerank import rerank
from src.citations import ts_url

def _fetch_chunks_by_ids(ids: list[str]) -> list[Chunk]:
    if not ids:
        return []
    db = SessionLocal()
    try:
        rows = db.execute(select(Chunk).where(Chunk.id.in_(ids))).scalars().all()
        by_id = {r.id: r for r in rows}
        return [by_id[i] for i in ids if i in by_id]
    finally:
        db.close()

def _fetch_titles(video_ids: list[str]) -> dict[str, str | None]:
    if not video_ids:
        return {}
    db = SessionLocal()
    try:
        rows = db.execute(select(Video).where(Video.id.in_(video_ids))).scalars().all()
        return {r.id: r.title for r in rows}
    finally:
        db.close()

def answer_question(question: str, namespace: str, summary: str = "", recent_turns: list[dict] | None = None, video_filter: list[str] | None = None):

    timings: dict[str, float] = {}
    cache_info: dict[str, bool] = {}

    t0 = time.perf_counter()

    # 1) rewrite (cached)
    # t = time.perf_counter()
    # rewritten, hit = rewrite_query(question, namespace=namespace)
    # timings["rewrite_ms"] = (time.perf_counter() - t) * 1000
    # cache_info["rewrite_hit"] = hit
    recent_turns = recent_turns or []
    rewritten, hit = rewrite_query(question, namespace=namespace, summary=summary or "", recent_turns=recent_turns)


    # 2) query embedding (cached)
    t = time.perf_counter()
    emb = OpenAIEmbeddings(model=EMBED_MODEL, api_key=OPENAI_API_KEY)

    qkey = f"qembed:{EMBED_MODEL}:{sha1(rewritten)}"
    qcached = get_json(qkey)
    if qcached:
        qvec = qcached["vec"]
        cache_info["qembed_hit"] = True
    else:
        qvec = emb.embed_query(rewritten)
        set_json(qkey, {"vec": qvec}, ttl_seconds=30 * 24 * 3600)
        cache_info["qembed_hit"] = False

    timings["embed_query_ms"] = (time.perf_counter() - t) * 1000

    # 3) pinecone retrieve (cached short TTL)
    t = time.perf_counter()
    dim = len(emb.embed_query("dimension probe"))
    index = ensure_index(dimension=dim)

    flt = {"video_id": {"$in": video_filter}} if video_filter else None
    filter_hash = sha1(json.dumps(flt, sort_keys=True)) if flt else "nofilter"
    rkey = f"retr:{namespace}:{sha1(rewritten)}:{FETCH_K}:{filter_hash}"

    rcached = get_json(rkey)
    if rcached:
        matches = rcached["matches"]
        cache_info["retrieval_hit"] = True
    else:
        query_res = index.query(
            vector=qvec,
            top_k=FETCH_K,
            include_metadata=True,
            namespace=namespace,
            filter=flt
        )

        # Convert QueryResponse → plain JSON-safe dict
        matches = []
        for m in query_res.matches:
            matches.append({
                "id": m.id,
                "score": float(m.score),
                "metadata": dict(m.metadata) if m.metadata else {}
            })

        set_json(rkey, {"matches": matches}, ttl_seconds=15 * 60)
        cache_info["retrieval_hit"] = False


    timings["retrieve_ms"] = (time.perf_counter() - t) * 1000

    #matches = res.get("matches", []) if isinstance(res, dict) else []
    ids = [m["id"] for m in matches]

    # 4) fetch chunk texts from Postgres (can be cached per chunk, optional)
    t = time.perf_counter()
    # Simple per-chunk cache:
    chunk_texts = []
    chunk_objs = []
    misses = []
    for cid in ids:
        ckey = f"chunk:{cid}"
        c = get_json(ckey)
        if c:
            chunk_texts.append(c["text"])
        else:
            misses.append(cid)

    fetched = _fetch_chunks_by_ids(misses)
    by_id = {c.id: c for c in fetched}
    for cid in misses:
        c = by_id.get(cid)
        if c:
            set_json(f"chunk:{cid}", {"text": c.text, "video_id": c.video_id, "start": c.start, "end": c.end}, ttl_seconds=7*24*3600)
            chunk_texts.append(c.text)
        else:
            chunk_texts.append("")

    # we still need ordered Chunk objects for sources; fetch all (some from cache)
    # lightweight: fetch all missing objects from DB, and synthesize cached ones:
    cached_objs = []
    for cid in ids:
        ckey = f"chunk:{cid}"
        c = get_json(ckey)
        if c and c.get("video_id") is not None:
            cached_objs.append(Chunk(id=cid, video_id=c["video_id"], start=c["start"], end=c["end"], text=c["text"]))
        else:
            cached_objs.append(None)

    # if any None remains, fill from DB:
    need = [ids[i] for i, obj in enumerate(cached_objs) if obj is None]
    fill = _fetch_chunks_by_ids(need)
    fill_map = {c.id: c for c in fill}
    for i, obj in enumerate(cached_objs):
        if obj is None:
            cached_objs[i] = fill_map.get(ids[i])

    chunk_objs = [c for c in cached_objs if c is not None]

    timings["db_fetch_ms"] = (time.perf_counter() - t) * 1000

    # 5) rerank (LLM)
    t = time.perf_counter()
    candidate_strings = []
    for c in chunk_objs:
        candidate_strings.append(f"{c.video_id} @ {c.start}s\n{c.text}")

    keep_idx = rerank(question, candidate_strings, top_k=min(RERANK_TOP_N, TOP_K))
    reranked = [chunk_objs[i] for i in keep_idx if 0 <= i < len(chunk_objs)]
    reranked = reranked[:TOP_K] if reranked else chunk_objs[:TOP_K]
    timings["rerank_ms"] = (time.perf_counter() - t) * 1000

    # 6) titles
    t = time.perf_counter()
    vids = list({c.video_id for c in reranked})
    titles = _fetch_titles(vids)
    timings["titles_ms"] = (time.perf_counter() - t) * 1000

    # 7) generation
    t = time.perf_counter()
    contexts = []
    sources = []
    for c in reranked:
        title = titles.get(c.video_id) or f"YouTube {c.video_id}"
        contexts.append(f"[{title} | {c.video_id} | {c.start}s]\n{c.text}")
        sources.append({
            "video_id": c.video_id,
            "title": title,
            "start": c.start,
            "end": c.end,
            "url": ts_url(c.video_id, c.start),
        })

    prompt = (
        "You are a transcript-grounded assistant.\n"
        "Use ONLY the provided context.\n"
        "If context is insufficient, say: \"I don't know based on the video.\" \n"
        "Cite sources inline like (VIDEO_ID @ start_seconds).\n\n"
        f"Question: {question}\n\n"
        "Context:\n" + "\n\n---\n\n".join(contexts)
    )

    llm = ChatOpenAI(model=CHAT_MODEL, api_key=OPENAI_API_KEY, temperature=0)
    answer = llm.invoke(prompt).content.strip()
    timings["generate_ms"] = (time.perf_counter() - t) * 1000

    timings["total_ms"] = (time.perf_counter() - t0) * 1000

    return {
        "rewritten_query": rewritten,
        "answer": answer,
        "sources": sources,
        "timings": timings,
        "cache": cache_info,
        "retrieved_candidates": len(ids),
        "used_context": len(reranked),
        "contexts_used": contexts,
    }
