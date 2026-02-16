# 🎥 YouTube RAG Chatbot  
### Production-Grade Retrieval-Augmented Generation System with Memory

🚀 **Live Demo:**  
👉 https://youtube-multi-video-playlist-rag-be5fqxqp9w7rvcffmlp3dk.streamlit.app/

---

## 🔥 Overview

This project is a **production-style Retrieval-Augmented Generation (RAG) system** that allows users to chat with multiple YouTube videos or entire playlists using a conversational interface.

Unlike demo RAG apps, this system is built with:

- Scalable vector search
- Persistent relational storage
- Distributed caching
- Query rewriting
- Reranking
- Deduplication
- Evaluation harness
- Latency tracking
- Memory-aware chatbot behavior
- Cloud deployment

It demonstrates real-world AI system design beyond notebooks and toy examples.

---

# 🧠 What Problem Does This Solve?

YouTube videos are long and difficult to search precisely.

This system enables users to:

- Ingest multiple videos or playlists
- Ask natural language questions
- Get timestamped citations
- Continue conversations with memory
- Reduce hallucinations via grounding
- Measure retrieval quality
- Deploy publicly

---
# 🏗️ System Architecture
````

User (Streamlit Chat UI)
↓
Conversation Summary Memory
↓
Query Rewriting (context-aware)
↓
Embedding (cached)
↓
Pinecone Vector Search
↓
PostgreSQL (source-of-truth chunks)
↓
LLM Reranking
↓
Answer Generation (grounded + citations)

````

---

# 🗂️ Storage Design

| Component      | Responsibility |
|---------------|---------------|
| **Pinecone**  | Stores embeddings + minimal metadata |
| **PostgreSQL**| Stores full chunk text + video metadata |
| **Redis**     | Caches rewrite / embeddings / retrieval |
| **Streamlit** | UI + session memory |



# ✨ Key Features

## ✅ Multi-Video & Playlist Ingestion
- Automatic chunking with overlap
- Timestamp-aware metadata
- Idempotent ingestion (dedup safe)
- SHA1-based chunk identity
- Safe re-indexing

## ✅ Conversation Memory
- Rolling summary (token-efficient)
- Multi-turn contextual dialogue
- Follow-up resolution:
  - “What about that part?”
  - “Explain that in more detail”

## ✅ Production Caching (Redis)
- Query rewrite cache
- Embedding cache
- Retrieval cache
- Chunk text cache
- Fail-open design (app runs even if cache fails)

## ✅ Reranking Layer
Improves precision by re-evaluating top-K candidates before final generation.

## ✅ Deduplication
- Stable chunk IDs
- DB constraints
- Safe re-ingestion

## ✅ Evaluation Harness
- RAGAS metrics
- Faithfulness scoring
- Retrieval quality measurement
- Latency logging per stage

## ✅ Observability
Per-stage latency tracking:
- Rewrite
- Embedding
- Retrieval
- DB fetch
- Rerank
- Generation



# 📊 Why This Is Not a Demo RAG

Most RAG projects:
- Store full text in vector DB
- No deduplication
- No caching
- No memory
- No evaluation
- No deployment

This project demonstrates:

- Proper separation of vector store vs source-of-truth database
- Scalable architecture
- Cloud-ready design
- Caching strategy
- Evaluation mindset
- Token-efficient memory handling
- Production resilience

---

# 🛠️ Tech Stack

- **Python**
- **Streamlit**
- **OpenAI API**
- **Pinecone**
- **PostgreSQL (Neon)**
- **Redis (Upstash)**
- **LangChain**
- **RAGAS**
- **Docker (local dev)**


# 📂 Project Structure

````
youtube-multi-video-playlist-rag/
├─ app.py
├─ requirements.txt
├─ docker-compose.yml
├─ README.md
└─ src/
  ├─ config.py
  ├─ cache.py
  ├─ db.py
  ├─ models.py
  ├─ ingest.py
  ├─ retrieve.py
  ├─ rewrite.py
  ├─ rerank.py
  ├─ memory.py
└─ eval/
````

# 🌍 Deployment

Deployed on:

- **Streamlit Community Cloud**
- **Neon (Postgres)**
- **Upstash (Redis)**
- **Pinecone (Vector DB)**

The system runs fully in the cloud using managed services.


# 🔍 Example Use Cases

- “What are the main themes discussed?”
- “Explain the concept mentioned at the beginning.”
- “What does the speaker say about embeddings?”
- “Compare what two videos say about scalability.”
- “Summarize the key takeaways.”


# 📈 Future Improvements

- Hybrid retrieval (BM25 + vector)
- MMR pre-reranking
- Persistent chat sessions
- User authentication
- Cost logging dashboard
- Structured JSON outputs


# 👨‍💻 About the Author

**Rishi Bethi**  
MSc AI & Automation  
AI Engineer focused on production-grade LLM systems


# ⭐ Why This Project Matters

This project demonstrates:

- AI system architecture
- Retrieval engineering
- Production design thinking
- Data modeling
- Caching strategies
- Evaluation-first mindset
- Deployment capability

It reflects real-world AI engineering practices rather than experimental notebooks.

