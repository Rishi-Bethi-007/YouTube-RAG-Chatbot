import json
from pathlib import Path
import pandas as pd

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from src.retrieve import answer_question
from src.config import NAMESPACE, CHAT_MODEL, EMBED_MODEL, OPENAI_API_KEY

TESTSET_PATH = Path("data/eval/testset.json")


def main():
    if not TESTSET_PATH.exists():
        print("❌ testset.json not found:", TESTSET_PATH)
        return

    tests = json.loads(TESTSET_PATH.read_text(encoding="utf-8"))
    if not tests:
        print("❌ testset.json is empty. Add at least 3–5 questions.")
        return

    rows = []
    print(f"\nRunning evaluation on {len(tests)} questions...\n")

    for i, t in enumerate(tests, 1):
        q = t["question"]
        gt = t.get("ground_truth", "")

        print(f"[{i}/{len(tests)}] Question: {q}")
        out = answer_question(q, namespace=NAMESPACE)

        # Use the actual chunk texts used in generation
        contexts = out.get("contexts_used", [])

        rows.append({
            "question": q,
            "answer": out["answer"],
            "contexts": contexts,
            "ground_truth": gt,
            "total_ms": out["timings"].get("total_ms"),
            "rewrite_ms": out["timings"].get("rewrite_ms"),
            "embed_query_ms": out["timings"].get("embed_query_ms"),
            "retrieve_ms": out["timings"].get("retrieve_ms"),
            "db_fetch_ms": out["timings"].get("db_fetch_ms"),
            "rerank_ms": out["timings"].get("rerank_ms"),
            "generate_ms": out["timings"].get("generate_ms"),
        })

    df = pd.DataFrame(rows)

    print("\n=== Latency summary (ms) ===")
    cols = ["total_ms","rewrite_ms","embed_query_ms","retrieve_ms","db_fetch_ms","rerank_ms","generate_ms"]
    print(df[cols].describe())

    # Build dataset for RAGAS
    ds = Dataset.from_dict({
        "question": df["question"].tolist(),
        "answer": df["answer"].tolist(),
        "contexts": df["contexts"].tolist(),
        "ground_truth": df["ground_truth"].tolist(),
    })

    # ✅ Provide evaluation LLM + embeddings explicitly
    eval_llm = LangchainLLMWrapper(
        ChatOpenAI(
            model=CHAT_MODEL,
            api_key=OPENAI_API_KEY,
            temperature=0,
        )
    )

    eval_embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(
            model=EMBED_MODEL,
            api_key=OPENAI_API_KEY,
        )
    )

    print("\nRunning RAGAS metrics...")
    result = evaluate(
        ds,
        metrics=[faithfulness, answer_relevancy],
        llm=eval_llm,
        embeddings=eval_embeddings,
    )

    print("\n=== RAGAS Results ===")
    print(result)


if __name__ == "__main__":
    main()
