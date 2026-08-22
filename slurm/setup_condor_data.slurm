#!/bin/bash
#SBATCH --job-name=condor_setup
#SBATCH --partition=preempt
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=300G
#SBATCH --time=48:00:00
#SBATCH --output=/home/jgibson2/projects/adaptive_rag/log/slurm_logs/%j_condor_setup.out
#SBATCH --error=/home/jgibson2/projects/adaptive_rag/log/slurm_logs/%j_condor_setup.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dalton.edu02@gmail.com

# Rebuild everything that was lost when aqawflow/ was deleted:
#   1. Qwen2.5-7B-Instruct  → condor/checkpoints/sft/qwen2.5_7b_instruct
#   2. Qwen2.5-14B-Instruct → condor/checkpoints/sft/qwen2.5_14b_instruct
#   3. DPR Wikipedia corpus  → condor/retriever/psgs_w100.tsv
#   4. E5-base-v2 model     → condor/retriever/intfloat/e5-base-v2
#   5. E5 FAISS index       → condor/retriever/wikipedia.e5
#   6. HotpotQA test split  → condor/data/hotpot_qa/test_996.parquet
#   7. NQ test split        → condor/data/nq/test_996.parquet

export PYTHONUNBUFFERED=1

source ~/.bashrc
source /home/jgibson2/projects/adaptive_rag/.venv/bin/activate

PROJECT_ROOT=/home/jgibson2/projects/adaptive_rag
BASELINE_ROOT=${PROJECT_ROOT}/baseline
export PYTHONPATH="${PROJECT_ROOT}:${BASELINE_ROOT}:${PYTHONPATH:-}"

CONDOR_ROOT=/data/user_data/jgibson2/condor
SFT_DIR=${CONDOR_ROOT}/checkpoints/sft
RETRIEVER_DIR=${CONDOR_ROOT}/retriever
DATA_DIR=${CONDOR_ROOT}/data

export HF_HOME=/data/user_data/jgibson2/hf_cache

echo "[SETUP] Job ID : ${SLURM_JOB_ID}"
echo "[SETUP] Node   : $(hostname)"
echo "[SETUP] GPU    : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "[SETUP] Start  : $(date)"
echo ""

die() { echo "[SETUP] FATAL: $*"; exit 1; }

# ── 1. Download Qwen2.5-7B-Instruct ──────────────────────────────────────────
DEST_7B=${SFT_DIR}/qwen2.5_7b_instruct
echo "[SETUP] === Step 1: Qwen2.5-7B-Instruct ==="
if [ -f "${DEST_7B}/config.json" ]; then
    echo "[SETUP]   already present, skipping"
else
    mkdir -p "${DEST_7B}"
    python - <<'EOF'
import os
from huggingface_hub import snapshot_download
dest = "/data/user_data/jgibson2/condor/checkpoints/sft/qwen2.5_7b_instruct"
print(f"  Downloading Qwen2.5-7B-Instruct → {dest}")
snapshot_download(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    local_dir=dest,
    local_dir_use_symlinks=False,
    ignore_patterns=["*.pt", "original/*"],
)
print("  Done.")
EOF
    [ -f "${DEST_7B}/config.json" ] || die "7B download failed"
fi
echo "[SETUP]   7B checkpoint ready ($(du -sh ${DEST_7B} | cut -f1))"
echo ""

# ── 2. Download Qwen2.5-14B-Instruct ─────────────────────────────────────────
DEST_14B=${SFT_DIR}/qwen2.5_14b_instruct
echo "[SETUP] === Step 2: Qwen2.5-14B-Instruct ==="
if [ -f "${DEST_14B}/config.json" ]; then
    echo "[SETUP]   already present, skipping"
else
    mkdir -p "${DEST_14B}"
    python - <<'EOF'
import os
from huggingface_hub import snapshot_download
dest = "/data/user_data/jgibson2/condor/checkpoints/sft/qwen2.5_14b_instruct"
print(f"  Downloading Qwen2.5-14B-Instruct → {dest}")
snapshot_download(
    repo_id="Qwen/Qwen2.5-14B-Instruct",
    local_dir=dest,
    local_dir_use_symlinks=False,
    ignore_patterns=["*.pt", "original/*"],
)
print("  Done.")
EOF
    [ -f "${DEST_14B}/config.json" ] || die "14B download failed"
fi
echo "[SETUP]   14B checkpoint ready ($(du -sh ${DEST_14B} | cut -f1))"
echo ""

# ── 3. Download intfloat/e5-base-v2 ──────────────────────────────────────────
E5_DIR=${RETRIEVER_DIR}/intfloat/e5-base-v2
echo "[SETUP] === Step 3: intfloat/e5-base-v2 ==="
if [ -f "${E5_DIR}/config.json" ]; then
    echo "[SETUP]   already present, skipping"
else
    mkdir -p "${E5_DIR}"
    python - <<'EOF'
from huggingface_hub import snapshot_download
dest = "/data/user_data/jgibson2/condor/retriever/intfloat/e5-base-v2"
print(f"  Downloading e5-base-v2 → {dest}")
snapshot_download(
    repo_id="intfloat/e5-base-v2",
    local_dir=dest,
    local_dir_use_symlinks=False,
)
print("  Done.")
EOF
    [ -f "${E5_DIR}/config.json" ] || die "e5-base-v2 download failed"
fi
echo "[SETUP]   E5 model ready"
echo ""

# ── 4. Download DPR Wikipedia passages (psgs_w100.tsv) ───────────────────────
PSGS_GZ=${RETRIEVER_DIR}/psgs_w100.tsv.gz
PSGS_TSV=${RETRIEVER_DIR}/psgs_w100.tsv
echo "[SETUP] === Step 4: psgs_w100.tsv (~14 GB) ==="
mkdir -p "${RETRIEVER_DIR}"
if [ -f "${PSGS_TSV}" ]; then
    echo "[SETUP]   already present ($(du -sh ${PSGS_TSV} | cut -f1)), skipping"
else
    if [ ! -f "${PSGS_GZ}" ]; then
        echo "[SETUP]   downloading from Facebook DPR..."
        wget -q --show-progress -O "${PSGS_GZ}" \
            https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz \
            || die "psgs_w100.tsv.gz download failed"
    fi
    echo "[SETUP]   decompressing..."
    gunzip -k "${PSGS_GZ}" || die "gunzip failed"
    rm -f "${PSGS_GZ}"
    echo "[SETUP]   psgs_w100.tsv ready ($(du -sh ${PSGS_TSV} | cut -f1))"
fi
echo ""

# ── 5. Build FAISS E5 index (wikipedia.e5) ────────────────────────────────────
FAISS_INDEX=${RETRIEVER_DIR}/wikipedia.e5
echo "[SETUP] === Step 5: Build FAISS E5 index ==="
if [ -f "${FAISS_INDEX}" ]; then
    FAISS_SIZE=$(du -sh "${FAISS_INDEX}" | cut -f1)
    echo "[SETUP]   already present (${FAISS_SIZE}), skipping"
else
    echo "[SETUP]   building wikipedia.e5 (GPU indexing — may take 8-12 hours at batch=64)..."
    echo "[SETUP]   Start: $(date)"
    export RETRIEVER_ROOT=${RETRIEVER_DIR}
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    python - <<'EOF'
import os
import numpy as np
import pandas as pd
import faiss
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

RETRIEVER_ROOT = os.environ["RETRIEVER_ROOT"]
PSG_PATH   = os.path.join(RETRIEVER_ROOT, "psgs_w100.tsv")
MODEL_PATH = os.path.join(RETRIEVER_ROOT, "intfloat/e5-base-v2")
SAVE_PATH  = os.path.join(RETRIEVER_ROOT, "wikipedia.e5")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  device: {device}")

print("  loading e5-base-v2...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model     = AutoModel.from_pretrained(MODEL_PATH).to(device)
model.eval()

# dtype=str avoids StringDtype NaN float objects that break concatenation
print(f"  loading psgs_w100.tsv ({PSG_PATH})...")
df = pd.read_csv(PSG_PATH, sep="\t", dtype=str)
df = df.fillna("")
titles = df["title"].tolist()
texts  = df["text"].tolist()
n = len(df)
print(f"  passages: {n}")

DIM              = 768
BATCH_SIZE       = 64       # 2048 OOM'd on L40S; 64×256tok uses ~600MB activations
CHECKPOINT_EVERY = 5000     # save partial index every ~320K passages (~7 min)
CKPT_IDX  = SAVE_PATH + ".partial"
CKPT_META = SAVE_PATH + ".partial.meta"

def mean_pool(token_embs, mask):
    token_embs = token_embs.masked_fill(~mask[..., None].bool(), 0.0)
    return token_embs.sum(dim=1) / mask.sum(dim=1)[..., None]

def embed_batch(sentences):
    inputs = tokenizer(sentences, padding=True, truncation=True,
                       max_length=256, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs)
    emb = mean_pool(out[0], inputs["attention_mask"])
    emb = emb / emb.norm(dim=1, keepdim=True)
    torch.cuda.empty_cache()
    return emb.cpu().numpy().astype(np.float32)

# Resume from checkpoint if one exists
start_batch = 0
if os.path.exists(CKPT_IDX) and os.path.exists(CKPT_META):
    with open(CKPT_META) as f:
        start_batch = int(f.read().strip())
    index = faiss.read_index(CKPT_IDX)
    print(f"  Resuming from batch {start_batch} "
          f"({index.ntotal/1e6:.2f}M vectors already indexed)")
else:
    index = faiss.IndexFlatIP(DIM)

total_batches = (n + BATCH_SIZE - 1) // BATCH_SIZE

for step, psg_start in enumerate(
        tqdm(range(start_batch * BATCH_SIZE, n, BATCH_SIZE),
             desc="indexing", initial=start_batch, total=total_batches)):
    psg_end  = min(psg_start + BATCH_SIZE, n)
    batch    = [titles[i] + "\t" + texts[i] for i in range(psg_start, psg_end)]
    embs     = embed_batch(batch)
    index.add(embs)

    cur_batch = start_batch + step + 1
    if cur_batch % CHECKPOINT_EVERY == 0:
        faiss.write_index(index, CKPT_IDX)
        with open(CKPT_META, "w") as f:
            f.write(str(cur_batch))
        print(f"\n  [ckpt] batch {cur_batch}/{total_batches}  "
              f"{index.ntotal/1e6:.2f}M vectors")

faiss.write_index(index, SAVE_PATH)
# Clean up partial checkpoint once final index is written
for p in [CKPT_IDX, CKPT_META]:
    if os.path.exists(p):
        os.remove(p)
print(f"  FAISS index saved → {SAVE_PATH}  ({index.ntotal} vectors)")
EOF
    [ -f "${FAISS_INDEX}" ] || die "FAISS index build failed"
    echo "[SETUP]   End: $(date)  size=$(du -sh ${FAISS_INDEX} | cut -f1)"
fi
echo ""

# ── 6. Download / generate test splits ───────────────────────────────────────
echo "[SETUP] === Step 6: test splits ==="
mkdir -p ${DATA_DIR}/hotpot_qa ${DATA_DIR}/nq

# HotpotQA test_996.parquet
if [ -f "${DATA_DIR}/hotpot_qa/test_996.parquet" ]; then
    echo "[SETUP]   hotpot_qa/test_996.parquet already present, skipping"
else
    echo "[SETUP]   building hotpot_qa/test_996.parquet from HuggingFace..."
    python - <<'EOF'
import sys
sys.path.insert(0, "/home/jgibson2/projects/adaptive_rag")
sys.path.insert(0, "/home/jgibson2/projects/adaptive_rag/baseline")
import os, json
from datasets import load_dataset, Dataset

try:
    from condor.mechanism_router import classify_qtype
    def oracle_mech(q):
        qtype = classify_qtype(q)
        return {0: 1, 1: 2, 2: 3, 3: 4}.get(qtype, 2)
except ImportError:
    def oracle_mech(q):
        return 2  # HotpotQA is multi-hop → m2

DATA_SOURCE = "hotpotqa/hotpot_qa"
print("  Downloading hotpotqa/hotpot_qa (fullwiki)...")
raw = load_dataset(DATA_SOURCE, "fullwiki")

# Use validation split as test set (matches original 996-sample convention)
test_raw = raw["validation"]

def process_fn(example, idx):
    q  = example["question"]
    # HotpotQA answer is a string
    a  = example["answer"]
    return {
        "data_source": DATA_SOURCE,
        "prompt": [{"role": "user", "content": q}],
        "ability": "QA",
        "reward_model": {"style": "rule", "ground_truth": a},
        "extra_info": {
            "split": "test",
            "index": idx,
            "answer": a,
            "question": q,
            "mechanism_label": oracle_mech(q),
        },
    }

test_ds = test_raw.select(range(min(996, len(test_raw)))).map(
    process_fn, with_indices=True, remove_columns=test_raw.column_names
)
out = "/data/user_data/jgibson2/condor/data/hotpot_qa/test_996.parquet"
test_ds.to_parquet(out)
print(f"  Saved {len(test_ds)} rows → {out}")
EOF
    [ -f "${DATA_DIR}/hotpot_qa/test_996.parquet" ] || die "hotpot_qa test split failed"
fi

# NQ test_996.parquet
if [ -f "${DATA_DIR}/nq/test_996.parquet" ]; then
    echo "[SETUP]   nq/test_996.parquet already present, skipping"
else
    echo "[SETUP]   building nq/test_996.parquet from HuggingFace..."
    python - <<'EOF'
import sys
sys.path.insert(0, "/home/jgibson2/projects/adaptive_rag")
sys.path.insert(0, "/home/jgibson2/projects/adaptive_rag/baseline")
from datasets import load_dataset

try:
    from condor.mechanism_router import classify_qtype
    def oracle_mech(q):
        qtype = classify_qtype(q)
        return {0: 1, 1: 2, 2: 3, 3: 4}.get(qtype, 1)
except ImportError:
    def oracle_mech(q):
        return 1  # NQ is factoid → m1

DATA_SOURCE = "google-research-datasets/nq_open"
print("  Downloading nq_open validation...")
raw = load_dataset(DATA_SOURCE)
test_raw = raw["validation"]

def process_fn(example, idx):
    q  = example["question"]
    a  = example["answer"][0] if isinstance(example["answer"], list) else example["answer"]
    return {
        "data_source": DATA_SOURCE,
        "prompt": [{"role": "user", "content": q}],
        "ability": "QA",
        "reward_model": {"style": "rule", "ground_truth": a},
        "extra_info": {
            "split": "test",
            "index": idx,
            "answer": a,
            "question": q,
            "mechanism_label": oracle_mech(q),
        },
    }

test_ds = test_raw.select(range(min(996, len(test_raw)))).map(
    process_fn, with_indices=True, remove_columns=test_raw.column_names
)
out = "/data/user_data/jgibson2/condor/data/nq/test_996.parquet"
test_ds.to_parquet(out)
print(f"  Saved {len(test_ds)} rows → {out}")
EOF
    [ -f "${DATA_DIR}/nq/test_996.parquet" ] || die "nq test split failed"
fi
echo ""

# ── Summary ───────────────────────────────────────────────────────────────────
echo "[SETUP] === All steps complete ==="
echo "[SETUP] End: $(date)"
echo ""
echo "  7B model    : $(du -sh ${SFT_DIR}/qwen2.5_7b_instruct | cut -f1)"
echo "  14B model   : $(du -sh ${SFT_DIR}/qwen2.5_14b_instruct | cut -f1)"
echo "  retriever   : $(du -sh ${RETRIEVER_DIR} | cut -f1)"
echo "  hotpot test : $(du -sh ${DATA_DIR}/hotpot_qa/test_996.parquet | cut -f1)"
echo "  nq test     : $(du -sh ${DATA_DIR}/nq/test_996.parquet | cut -f1)"
echo ""
echo "[SETUP] You can now submit: sbatch slurm/train_condor_hotpot.slurm"
