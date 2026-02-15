# 📺 YouTube Multi-Video / Playlist RAG  
**Production-style Retrieval-Augmented Chatbot with Memory**

Chat with multiple YouTube videos (or entire playlists) using a scalable RAG architecture built with:

- **Streamlit** (ChatGPT-style UI)
- **Pinecone** (Vector database)
- **PostgreSQL** (Source of truth)
- **Redis** (Caching layer)
- **OpenAI** (Embeddings + LLM)
- **Reranking + Dedup + Evaluation (RAGAS)**

---

## 🚀 What This Project Solves

YouTube videos are long and difficult to search precisely.

This system allows you to:

- Ingest **multiple videos or playlists**
- Ask natural language questions
- Get **timestamped citations**
- Have **multi-turn conversations with memory**
- Evaluate retrieval quality
- Log latency per stage
- Deploy publicly

---

# 🏗️ Architecture

User → Streamlit Chat UI
↓
Query Rewrite (memory-aware)
↓
Embedding (cached)
↓
Pinecone (vector search)
↓
Postgres (fetch chunk text)
↓
Rerank (LLM-based)
↓
Answer Generation (chat-style + citations)


### Storage Design

| Component     | Responsibility |
|--------------|---------------|
| Pinecone     | Stores vectors + minimal metadata |
| PostgreSQL   | Stores full chunk text + video metadata |
| Redis        | Caches rewrite / embeddings / retrieval / chunk text |
| Streamlit    | UI + session memory |

---

# ✨ Features

## ✅ Multi-Video / Playlist Ingestion
- Automatic chunking
- Timestamp-aware segments
- Idempotent ingestion (dedup safe)
- Video-level ingestion tracking

## ✅ Production Caching (Redis)
- Query rewrite cache
- Embedding cache
- Retrieval cache
- Chunk text cache

## ✅ Reranking
LLM-based reranker improves precision by selecting the best context from top-K retrieved candidates.

## ✅ Conversation Memory
- Maintains rolling summary
- Supports follow-ups like:
  - “What about that part?”
  - “Explain more about that concept”
- Token-efficient summarization

## ✅ Deduplication
- Chunk-level SHA1 IDs
- Unique DB constraints
- Ingestion logs prevent re-embedding

## ✅ Evaluation Harness
- RAGAS metrics:
  - Faithfulness
  - Answer relevancy
- Stage-level latency tracking

## ✅ Observability
- Per-stage timing:
  - Rewrite
  - Embedding
  - Retrieval
  - DB fetch
  - Rerank
  - Generation
- Cache hit tracking

---

# 📂 Project Structure



youtube-rag/
├─ app.py
├─ docker-compose.yml
├─ requirements.txt
├─ .env.example
├─ data/
│ ├─ transcripts/
│ └─ eval/testset.json
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
└─ eval/run_eval.py


---

# 🧠 How Memory Works

We maintain:

- Full chat in session
- A rolling **summary**
- Last N turns passed to:
  - Query rewriting
  - Answer generation

This enables contextual multi-turn dialogue without exploding token usage.

---

# 🛠️ Local Setup

## 1️⃣ Start Postgres + Redis

```bash
docker compose up -d

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Create .env

Copy .env.example and fill in:

OPENAI_API_KEY=
PINECONE_API_KEY=
DATABASE_URL=postgresql+psycopg2://rag:rag@localhost:5432/ragdb
REDIS_URL=redis://localhost:6379/0

4️⃣ Run the app
streamlit run app.py

🧪 Run Evaluation
python -m src.eval.run_eval


Outputs:

Latency summary

RAGAS metrics

🌍 Deployment (Free)
Recommended Stack

Streamlit Community Cloud

Neon (free Postgres)

Upstash (free Redis)

Pinecone free tier

Steps

Push repo to GitHub

Create Neon Postgres → copy DATABASE_URL

Create Upstash Redis → copy REDIS_URL

Deploy app on Streamlit Cloud

Add secrets in app settings

📊 Production Design Decisions
Why Postgres?

Vector DB is not source of truth.
Text storage belongs in relational DB for:

Re-indexing

Analytics

Versioning

Data governance

Why Redis?

LLM apps are latency-sensitive.
Redis reduces:

Embedding cost

Pinecone round trips

DB pressure

Why Reranking?

Vector similarity alone is noisy.
Reranking dramatically improves answer precision.

Why Summary Memory?

Full chat history grows tokens exponentially.
Summaries maintain context efficiently.

🔍 Example Questions

“What are the main themes discussed?”

“Explain the concept mentioned at the beginning.”

“What does the speaker say about scalability?”

“What about the part where he talks about embeddings?”

“Summarize the difference between FAISS and Pinecone.”

📈 Future Improvements

MMR before reranking

Hybrid search (BM25 + vector)

Persistent chat sessions

User authentication

Cost logging per query

Structured JSON answers
