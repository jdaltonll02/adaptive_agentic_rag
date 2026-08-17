"""Retrieval server for the Babel cluster.

Serves two endpoints to satisfy both callers:

  POST /search    (RetrievalAgent in qa_manager/BaseAgent.py)
      request:  {"questions": ["q1", ...], "N": 5}
      response: [{"question": "q1", "top_k_docs": ["title\ntext", ...]}, ...]

  POST /retrieve  (verl.tools.search_tool.SearchTool)
      request:  {"queries": ["q1", ...], "topk": 3, "return_scores": true}
      response: {"result": [[{"document": {"contents": "title\ntext"}, "score": 0.9}, ...], ...]}

  GET  /health    health check → {"status": "ok"}

Environment variables:
  RETRIEVER_ROOT   – directory with the FAISS index, model, and TSV
                     default: /data/user_data/jgibson2/condor/retriever
  RETRIEVAL_PORT   – port to listen on (default: 8000)
  RETRIEVAL_USE_GPU – 1 to use GPU FAISS (default 0 = CPU; avoids conflict
                      with the training job using the same GPU)
"""
import os
import time

import faiss
import numpy as np
import pandas as pd
import torch
from flask import Flask, jsonify, request
from threading import Lock
from transformers import AutoModel, AutoTokenizer

RETRIEVER_ROOT = os.environ.get(
    "RETRIEVER_ROOT", "/data/user_data/jgibson2/condor/retriever"
)
DOCS_PATH   = os.path.join(RETRIEVER_ROOT, "psgs_w100.tsv")
MODEL_PATH  = os.path.join(RETRIEVER_ROOT, "intfloat/e5-base-v2")
INDEX_PATH  = os.path.join(RETRIEVER_ROOT, "wikipedia.e5")
PORT        = int(os.environ.get("RETRIEVAL_PORT", 8000))
USE_GPU     = os.environ.get("RETRIEVAL_USE_GPU", "0") == "1"

# ── Load embedding model ──────────────────────────────────────────────────────
print("Loading retriever model...")
t0 = time.time()
device = torch.device("cuda:0" if (USE_GPU and torch.cuda.is_available()) else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model     = AutoModel.from_pretrained(MODEL_PATH).to(device)
model.eval()
print(f"  done ({time.time()-t0:.1f}s)  device={device}")

# ── Load FAISS index ──────────────────────────────────────────────────────────
print("Loading FAISS index (this may take several minutes for the 64 GB index)...")
t0 = time.time()
index = faiss.read_index(INDEX_PATH)
if USE_GPU and torch.cuda.is_available():
    res   = faiss.StandardGpuResources()
    index = faiss.index_cpu_to_gpu(res, 0, index)
    print(f"  loaded to GPU ({time.time()-t0:.1f}s)")
else:
    print(f"  loaded to CPU ({time.time()-t0:.1f}s)")

# ── Load corpus TSV ───────────────────────────────────────────────────────────
print("Loading corpus TSV...")
t0 = time.time()
df = pd.read_csv(DOCS_PATH, sep="\t")
print(f"  done ({time.time()-t0:.1f}s)  rows={len(df)}")

app  = Flask(__name__)
lock = Lock()


# ── Embedding helper ──────────────────────────────────────────────────────────

def _mean_pool(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.0)
    return token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]


def _embed(sentences):
    # e5 models expect "query: <text>" prefix for queries
    prefixed = ["query: " + s for s in sentences]
    inputs = tokenizer(prefixed, padding=True, truncation=True,
                       max_length=512, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs)
    emb = _mean_pool(out[0], inputs["attention_mask"])
    # Normalise for cosine similarity
    emb = emb / emb.norm(dim=1, keepdim=True)
    return emb.cpu().numpy()


def _search(questions, n):
    """Search FAISS index. Returns list of lists of (title, text, score) tuples."""
    embs = _embed(questions)
    scores_batch, idxs_batch = index.search(embs.astype(np.float32), n)
    results = []
    for scores, idxs in zip(scores_batch, idxs_batch):
        docs = []
        for doc_id, score in zip(idxs, scores):
            if doc_id < 0 or doc_id >= len(df):
                continue
            row   = df.iloc[doc_id]
            title = str(row.get("title", ""))
            text  = str(row.get("text", ""))
            docs.append((title, text, float(score)))
        results.append(docs)
    return results


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/search", methods=["POST"])
def search_endpoint():
    """RetrievalAgent format."""
    with lock:
        data      = request.get_json(force=True)
        questions = data.get("questions", [])
        n         = int(data.get("N", 5))
        if not questions:
            return jsonify([])

        batch = _search(questions, n)
        response = []
        for q, docs in zip(questions, batch):
            top_k_docs = [f"{title}\n{text}" for title, text, _score in docs]
            response.append({"question": q, "top_k_docs": top_k_docs})
        return jsonify(response)


@app.route("/retrieve", methods=["POST"])
def retrieve_endpoint():
    """SearchTool (verl.tools.search_tool) format."""
    with lock:
        data         = request.get_json(force=True)
        queries      = data.get("queries", [])
        topk         = int(data.get("topk", 3))
        return_scores = data.get("return_scores", True)

        if not queries:
            return jsonify({"result": []})

        batch = _search(queries, topk)
        result = []
        for docs in batch:
            per_query = []
            for title, text, score in docs:
                contents = f"{title}\n{text}"
                entry = {"document": {"contents": contents}}
                if return_scores:
                    entry["score"] = score
                per_query.append(entry)
            result.append(per_query)
        return jsonify({"result": result})


if __name__ == "__main__":
    print(f"\nStarting retrieval server on port {PORT} (GPU={USE_GPU})")
    # threaded=True: FAISS CPU search and E5 eval are thread-safe with torch.no_grad()
    app.run(host="0.0.0.0", port=PORT, threaded=True)
