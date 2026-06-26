#!/usr/bin/env python3
"""Dataroom sidecar: local jina models over the UNZIPPED input dataroom (read-only).

Two local Jina models, no network/web tools:
  - jina-embeddings-v5-text-small  -> semantic search over the dataroom  (/search)
  - jina-reranker-v3               -> cross-encoder reranking          (/rerank)

NOTHING is indexed at boot. There is no precomputed index. The model decides, during the run,
whether it wants embedding search at all - and if so, over what scope and with what chunking.
/search embeds ON THE FLY: it reads the requested files (all dataroom files, or only the `paths`
the model names), chunks them (default size, or a `chunk_size` the model picks), embeds, and
returns the top matches. Within one run we memoize an embedded scope so a repeated identical
search does not re-embed, but that index is built lazily on first use, never ahead of time.

jina-embeddings-v5 retrieval is ASYMMETRIC: queries use the "query" prompt and passages
the "document" prompt, so cosine scores are calibrated.

Endpoints (POST JSON):
  /search  {query, k=8, paths?[], chunk_size?, chunk_overlap?}  -> top-k chunks (on-the-fly embed)
  /rerank  {query, documents[], top_n?}                          -> reranker-v3 ordering
  /stats   {}                                                    -> {files, file_count, models}
"""
import os, json, glob, hashlib, time
import numpy as np
from fastapi import FastAPI, Request
import urllib.request
import uvicorn

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Backend per model: "local" (self-hosted weights) or "api" (Jina cloud, https://api.jina.ai).
# This is invisible to the agent - the tools and their outputs are identical either way; only
# this env decides where the embed/rerank actually runs. Clean ablation axis: local vs api.
EMBED_BACKEND = os.environ.get("EMBED_BACKEND", "local").lower()
RERANK_BACKEND = os.environ.get("RERANK_BACKEND", "local").lower()
JINA_API_KEY = os.environ.get("JINA_API_KEY", "")
JINA_API_BASE = os.environ.get("JINA_API_BASE", "https://api.jina.ai/v1")
# Model ids differ by backend: local uses HF repo ids, api uses Jina model names.
API_EMBED_MODEL = os.environ.get("API_EMBED_MODEL", "jina-embeddings-v5-text-small")
API_RERANK_MODEL = os.environ.get("API_RERANK_MODEL", "jina-reranker-v3")

# Keep the LOCAL embedder/reranker off the GPU by default so the LLM owns VRAM (jina-v5
# base+LoRA otherwise OOM a tight card). Set EMBED_DEVICE=cuda when there is headroom.
EMBED_DEVICE = os.environ.get("EMBED_DEVICE", "cpu")
if EMBED_DEVICE.startswith("cpu"):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

DATAROOM_DIR = os.path.abspath(os.environ.get("DATAROOM_DIR", "dataroom"))
EMBED_MODEL = os.environ.get("EMBED_MODEL", "jinaai/jina-embeddings-v5-text-small")
RERANK_MODEL = os.environ.get("RERANK_MODEL", "jinaai/jina-reranker-v3")
EMBED_TASK = os.environ.get("EMBED_TASK", "retrieval")
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "1400"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "180"))

# Where /embed writes its jsonl output. This is the model's sandbox cwd (parent of dataroom/), so
# the model can read the file back with a plain relative path (cat / python). Falls back to CWD.
WORK_DIR = os.environ.get("WORK_DIR") or os.path.dirname(DATAROOM_DIR) or os.getcwd()


def _api_post(path: str, payload: dict) -> dict:
    req = urllib.request.Request(
        f"{JINA_API_BASE}/{path}",
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {JINA_API_KEY}",
                 "Content-Type": "application/json", "Accept": "application/json",
                 "User-Agent": "Mozilla/5.0"})  # CF blocks the default Python-urllib UA
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)

# Text-like files we will index. Binary/asset files are skipped (the agent can still `read`
# or `bash` them directly). Extendable via DATAROOM_GLOBS (comma-separated globs).
DEFAULT_GLOBS = (
    "**/*.md", "**/*.txt", "**/*.rst", "**/*.py", "**/*.js", "**/*.ts", "**/*.tsx",
    "**/*.json", "**/*.jsonl", "**/*.yaml", "**/*.yml", "**/*.toml", "**/*.csv",
    "**/*.tsv", "**/*.html", "**/*.htm", "**/*.xml", "**/*.tex", "**/*.c", "**/*.h",
    "**/*.cpp", "**/*.cc", "**/*.go", "**/*.rs", "**/*.java", "**/*.sql", "**/*.sh",
    "**/*.ipynb", "**/*.log", "**/*.cfg", "**/*.ini", "**/*.env",
    # Binary document formats: text is extracted on read (see _read_text). Without these the
    # dataroom (often ALL pdf/xlsx, e.g. regulatory filings) indexes 0 files -> every /search
    # returns scope_chunks:0 -> the model gets no grounding and hallucinates.
    "**/*.pdf", "**/*.xlsx", "**/*.xls", "**/*.docx",
)
_globs_env = os.environ.get("DATAROOM_GLOBS", "").strip()
INDEX_GLOBS = tuple(g.strip() for g in _globs_env.split(",") if g.strip()) or DEFAULT_GLOBS
# Binary docs pack more bytes per unit of content; allow a higher ceiling for them.
MAX_FILE_BYTES = int(os.environ.get("MAX_FILE_BYTES", str(4 * 1024 * 1024)))  # skip huge blobs
MAX_DOC_BYTES = int(os.environ.get("MAX_DOC_BYTES", str(32 * 1024 * 1024)))   # pdf/xlsx/docx ceiling
_DOC_EXTS = (".pdf", ".xlsx", ".xls", ".docx")

app = FastAPI()

_embed_model = None
_rerank_model = None
# Lazy per-scope index cache: key (paths tuple, chunk_size, overlap) -> {embs, meta}. Built only
# when /search is first called for that scope; nothing exists until the model asks. Cleared via
# nothing - it just lives for the process. No boot index, no on-disk precompute.
_scope_cache: dict = {}

# OPENCLAW_EMBED_CACHE: precise per-(text,role) embedding cache. Content-keyed (exact string match),
# so identical passages/queries across turns and across tools are embedded at most once per
# job process. All embedding paths funnel through _encode, so caching here accelerates embed,
# search, similarity, classify, cluster, deduplicate, and answer uniformly. Memory-bounded by
# unique text count in a single job (typically a few thousand chunks); fine for a sidecar.
_embed_cache: dict = {}
_embed_cache_stats = {"hits": 0, "misses": 0}


# --- models ------------------------------------------------------------------
# OPENCLAW_LOCAL_OFFLOAD: resolve target device, auto-falling back to CPU when CUDA is
# unavailable or has no usable free VRAM. EMBED_DEVICE=cuda asks for GPU; if the card is full
# (e.g. a colocated llama-server took all VRAM) we run on CPU instead of OOMing.
def _is_cuda_oom(e: Exception) -> bool:
    msg = str(e).lower()
    return ("out of memory" in msg or "cuda error" in msg or "cublas" in msg
            or "no kernel image" in msg or ("alloc" in msg and "cuda" in msg))


def _want_cuda() -> bool:
    if not EMBED_DEVICE.startswith("cuda"):
        return False
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        need = int(os.environ.get("MIN_FREE_VRAM_MB", "1500")) * 1024 * 1024
        free, _ = torch.cuda.mem_get_info()
        if free < need:
            print(f"[dataroom] CUDA free {free // 1048576}MB < need {need // 1048576}MB -> CPU",
                  flush=True)
            return False
        return True
    except Exception:
        return False


def embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        dev = "cuda" if _want_cuda() else "cpu"
        print(f"[dataroom] loading embed model {EMBED_MODEL} on {dev}", flush=True)
        try:
            _embed_model = SentenceTransformer(EMBED_MODEL, device=dev, trust_remote_code=True)
        except Exception as e:
            if dev == "cuda" and _is_cuda_oom(e):
                print(f"[dataroom] embed CUDA load failed ({e}); offloading to CPU", flush=True)
                _embed_model = SentenceTransformer(EMBED_MODEL, device="cpu", trust_remote_code=True)
            else:
                raise
    return _embed_model


def rerank_model():
    """Load the local reranker. Supports jina-reranker-v3 (AutoModel) and
    jina-reranker-v2-base-multilingual (AutoModelForSequenceClassification). Both expose
    .rerank(query, documents, top_n=...). Auto GPU->CPU offload on OOM / full card."""
    global _rerank_model
    if _rerank_model is None:
        from transformers import AutoModel, AutoModelForSequenceClassification
        print(f"[dataroom] loading rerank model {RERANK_MODEL}", flush=True)
        last = None
        m = None
        for loader in (AutoModel, AutoModelForSequenceClassification):
            try:
                m = loader.from_pretrained(RERANK_MODEL, dtype="auto", trust_remote_code=True)
                if hasattr(m, "rerank"):
                    break
                m = None  # loaded but wrong class (no .rerank); try the next loader
            except Exception as e:
                last = e
                continue
        if m is None:
            raise RuntimeError(f"could not load a reranker with .rerank() for {RERANK_MODEL}: {last}")
        m.eval()
        if _want_cuda():
            try:
                m = m.to("cuda")
            except Exception as e:
                if _is_cuda_oom(e):
                    print(f"[dataroom] rerank CUDA move failed ({e}); staying on CPU", flush=True)
                else:
                    raise
        _rerank_model = m
    return _rerank_model


def _encode_api(texts, role: str) -> np.ndarray:
    """Encode via Jina cloud embeddings API. Same v5 retrieval task, asymmetric query/passage."""
    task = "retrieval.query" if role == "query" else "retrieval.passage"
    out = []
    B = 128
    for i in range(0, len(texts), B):
        d = _api_post("embeddings", {"model": API_EMBED_MODEL, "task": task,
                                     "input": list(texts[i:i + B])})
        rows = sorted(d["data"], key=lambda r: r.get("index", 0))
        out.extend(r["embedding"] for r in rows)
    arr = np.asarray(out, dtype=np.float32)
    # API returns unnormalized; L2-normalize so cosine == dot, matching the local path.
    if arr.size:
        arr /= (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12)
    return arr


def _encode_raw(texts, role: str) -> np.ndarray:
    """Encode with the retrieval adapter + role-specific prompt (query vs document).

    Routes to the Jina cloud API when EMBED_BACKEND=api; otherwise the local model.
    Degrades gracefully if a build lacks the named prompts (older/odd v5 packaging)."""
    if EMBED_BACKEND == "api":
        return _encode_api(texts, role)
    m = embed_model()
    wants = ["query"] if role == "query" else ["document", "passage"]
    available = set((getattr(m, "prompts", None) or {}).keys())
    names = [n for n in wants if not available or n in available] or wants[:1]
    attempts = []
    for n in names:
        attempts.append({"task": EMBED_TASK, "prompt_name": n})
        attempts.append({"prompt_name": n})
    attempts.append({"task": EMBED_TASK})
    for kwargs in attempts:
        try:
            e = m.encode(texts, normalize_embeddings=True, **kwargs)
            return np.asarray(e, dtype=np.float32)
        except (TypeError, ValueError, KeyError):
            continue
    return np.asarray(m.encode(texts, normalize_embeddings=True), dtype=np.float32)


def _encode(texts, role: str) -> np.ndarray:
    """Cached front-end for _encode_raw. Returns embeddings for `texts` in order, embedding
    only the ones not already cached for this (text, role). Cache is per job process."""
    texts = list(texts)
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    miss_idx, miss_texts = [], []
    for i, t in enumerate(texts):
        if (role, t) not in _embed_cache:
            miss_idx.append(i)
            miss_texts.append(t)
    if miss_texts:
        _embed_cache_stats["misses"] += len(miss_texts)
        vecs = _encode_raw(miss_texts, role)
        vecs = np.asarray(vecs, dtype=np.float32)
        for j, i in enumerate(miss_idx):
            _embed_cache[(role, texts[i])] = vecs[j]
    _embed_cache_stats["hits"] += len(texts) - len(miss_texts)
    return np.asarray([_embed_cache[(role, t)] for t in texts], dtype=np.float32)


# --- dataroom indexing ---------------------------------------------------------
def chunk(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list:
    text = text.strip()
    if len(text) <= size:
        return [text] if text else []
    out, i = [], 0
    while i < len(text):
        out.append(text[i:i + size])
        i += size - overlap
    return out


def _dataroom_files() -> list:
    found = []
    seen = set()
    for g in INDEX_GLOBS:
        for ap in glob.glob(os.path.join(DATAROOM_DIR, g), recursive=True):
            if not os.path.isfile(ap) or ap in seen:
                continue
            # Binary docs get a higher size ceiling than plain text (they pack more bytes).
            cap = MAX_DOC_BYTES if ap.lower().endswith(_DOC_EXTS) else MAX_FILE_BYTES
            try:
                if os.path.getsize(ap) > cap:
                    continue
            except OSError:
                continue
            seen.add(ap)
            found.append(ap)
    return sorted(found)


# --- text extraction for binary docs -----------------------------------------
_extract_warned = set()


def _extract_pdf(ap: str) -> str:
    from pypdf import PdfReader
    reader = PdfReader(ap)
    parts = []
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            continue
    return "\n".join(parts)


def _extract_xlsx(ap: str) -> str:
    from openpyxl import load_workbook
    wb = load_workbook(ap, read_only=True, data_only=True)
    parts = []
    for ws in wb.worksheets:
        parts.append(f"# Sheet: {ws.title}")
        for row in ws.iter_rows(values_only=True):
            cells = [str(c) for c in row if c is not None]
            if cells:
                parts.append("\t".join(cells))
    wb.close()
    return "\n".join(parts)


def _extract_docx(ap: str) -> str:
    import docx
    d = docx.Document(ap)
    parts = [p.text for p in d.paragraphs if p.text]
    for tbl in d.tables:
        for row in tbl.rows:
            cells = [c.text for c in row.cells if c.text]
            if cells:
                parts.append("\t".join(cells))
    return "\n".join(parts)


def _read_text(ap: str) -> str:
    """Return indexable text for a dataroom file. Plain-text files are read as-is; binary
    documents (pdf/xlsx/docx) are extracted. Extraction failures degrade to empty (skipped),
    logged once per path, so one unreadable doc never kills the whole scope build."""
    low = ap.lower()
    try:
        if low.endswith(".pdf"):
            return _extract_pdf(ap)
        if low.endswith((".xlsx", ".xls")):
            return _extract_xlsx(ap)
        if low.endswith(".docx"):
            return _extract_docx(ap)
        return open(ap, errors="ignore").read()
    except Exception as e:
        if ap not in _extract_warned:
            _extract_warned.add(ap)
            print(f"WARN: text extraction failed for {ap}: {e}", flush=True)
        return ""


def _resolve_paths(paths) -> list:
    """Map caller-supplied paths to absolute dataroom files, guarding against escape.
    If no paths given, use every text-like dataroom file.

    Robust to how the model actually calls it (verified against real traces):
      - ABSOLUTE paths: the model echoes the absolute paths it saw in tool results
        (e.g. /.../work/dataroom/foo.md). os.path.join would discard DATAROOM_DIR for an
        absolute arg, so we strip a leading DATAROOM_DIR / "dataroom/" prefix and treat the
        remainder as dataroom-relative.
      - DIRECTORIES (incl. the dataroom root itself, e.g. paths=["."] or the abs dataroom dir):
        expand to every text-like file under that directory, instead of returning nothing.
      - Anything that still resolves to a single file is kept as-is.
    Any path that matches nothing is skipped; if NOTHING matches, fall back to the whole
    dataroom so a clumsy `paths` arg degrades to a full search rather than empty results."""
    if not paths:
        return _dataroom_files()
    base = os.path.realpath(DATAROOM_DIR)
    allowed = set(_dataroom_files())  # the canonical text-like file set (abspath, glob-based)
    allowed_real = {os.path.realpath(f): f for f in allowed}  # realpath -> canonical form
    out = []

    def _add(canonical):
        if canonical not in out:
            out.append(canonical)

    for p in paths:
        p = str(p).strip()
        if not p:
            continue
        # Normalize an absolute path that points inside (or at) the dataroom to dataroom-relative.
        if os.path.isabs(p):
            rp = os.path.realpath(p)
            if rp == base or rp.startswith(base + os.sep):
                p = os.path.relpath(rp, base)
            # else: leave as-is; the guard below rejects escapes.
        ap = os.path.realpath(os.path.join(DATAROOM_DIR, p))
        if not (ap == base or ap.startswith(base + os.sep)):
            continue  # escape attempt
        if os.path.isfile(ap):
            # Keep only files in the indexable set (canonical form), matched via realpath.
            canon = allowed_real.get(ap)
            if canon:
                _add(canon)
        elif os.path.isdir(ap):
            # Expand to every indexable file under this directory (or the whole dataroom if it
            # IS the dataroom root). Match via realpath to survive abspath/symlink differences.
            for fr, canon in allowed_real.items():
                if ap == base or fr == ap or fr.startswith(ap + os.sep):
                    _add(canon)
    # Clumsy/unmatched paths degrade to a full-dataroom search instead of empty results.
    if not out:
        return _dataroom_files()
    return out


def _build_scope(files: list, size: int, overlap: int):
    """Read+chunk+embed the given files NOW. Returns (embs (N,D) normalized, meta list)."""
    texts, meta = [], []
    for ap in files:
        raw = _read_text(ap)
        if not raw or not raw.strip():
            continue
        rel = os.path.relpath(ap, DATAROOM_DIR)
        for ci, c in enumerate(chunk(raw, size, overlap)):
            texts.append(c)
            meta.append({"path": rel, "chunk": ci, "text": c})
    embs = _encode(texts, role="passage") if texts else np.zeros((0, 0), dtype=np.float32)
    return embs, meta


# --- endpoints ---------------------------------------------------------------
@app.post("/search")
async def search(req: Request):
    """On-the-fly semantic search. Embeds the requested scope at call time (no boot index).

    body: {query, k=8, paths?[], chunk_size?, chunk_overlap?}
      - paths omitted  -> search the whole dataroom (model chose to search everything)
      - paths given    -> only those files (model scoped it down to a part)
      - chunk_size/overlap -> model controls granularity (defaults otherwise)
    """
    body = await req.json()
    q = body.get("query", "")
    if not q:
        return {"results": [], "scope_chunks": 0}
    k = int(body.get("k", 8))
    size = int(body.get("chunk_size", CHUNK_SIZE))
    overlap = int(body.get("chunk_overlap", CHUNK_OVERLAP))
    files = _resolve_paths(body.get("paths"))
    if not files:
        return {"results": [], "scope_chunks": 0}

    # Memoize this exact scope for the run so repeated identical searches don't re-embed.
    # Built lazily here on first use - never at boot.
    key = (tuple(sorted(os.path.relpath(f, DATAROOM_DIR) for f in files)), size, overlap)
    cached = _scope_cache.get(key)
    if cached is None:
        t0 = time.time()
        embs, meta = _build_scope(files, size, overlap)
        _scope_cache[key] = cached = {"embs": embs, "meta": meta}
        print(f"[dataroom] on-the-fly embed: {len(files)} files -> {len(meta)} chunks "
              f"(size={size}, overlap={overlap}) in {time.time()-t0:.1f}s", flush=True)
    embs, meta = cached["embs"], cached["meta"]
    if embs is None or embs.shape[0] == 0:
        return {"results": [], "scope_chunks": 0}

    qe = _encode([q], role="query")[0]
    sims = embs @ qe
    order = np.argsort(-sims)[:k]
    results = [{
        "path": meta[i]["path"],
        "chunk": int(meta[i]["chunk"]),
        "score": round(float(sims[i]), 4),
        "text": meta[i]["text"],
    } for i in order]
    return {"results": results, "scope_chunks": int(embs.shape[0]), "scope_files": len(files)}


@app.post("/answer")
async def answer(req: Request):
    """High-level two-stage QA retrieval: dense-retrieve a WIDE candidate set from the
    dataroom (embedding search), then cross-encoder RERANK to keep the few best supporting
    passages. Backs the answer_question tool. body: {query|question, k=5, paths?, fetch_k?}.
    Returns {results:[{path,chunk,score,text}]} reranked, plus scope stats."""
    body = await req.json()
    q = body.get("query") or body.get("question") or ""
    if not q:
        return {"results": [], "scope_chunks": 0}
    k = int(body.get("k", 5))
    fetch_k = int(body.get("fetch_k", max(20, k * 6)))
    size = int(body.get("chunk_size", CHUNK_SIZE))
    overlap = int(body.get("chunk_overlap", CHUNK_OVERLAP))
    files = _resolve_paths(body.get("paths"))
    if not files:
        return {"results": [], "scope_chunks": 0}
    key = (tuple(sorted(os.path.relpath(f, DATAROOM_DIR) for f in files)), size, overlap)
    cached = _scope_cache.get(key)
    if cached is None:
        embs, meta = _build_scope(files, size, overlap)
        _scope_cache[key] = cached = {"embs": embs, "meta": meta}
    embs, meta = cached["embs"], cached["meta"]
    if embs is None or embs.shape[0] == 0:
        return {"results": [], "scope_chunks": 0}
    # stage 1: dense retrieve wide
    qe = _encode([q], role="query")[0]
    sims = embs @ qe
    cand = np.argsort(-sims)[:fetch_k]
    cand_texts = [meta[i]["text"] for i in cand]
    # stage 2: cross-encoder rerank narrow
    ranked = _rerank_texts(q, cand_texts, top_n=k)
    results = []
    for r in ranked:
        ci = cand[int(r["index"])]
        results.append({"path": meta[ci]["path"], "chunk": int(meta[ci]["chunk"]),
                        "score": round(float(r["score"]), 4), "text": meta[ci]["text"]})
    return {"results": results, "scope_chunks": int(embs.shape[0]), "scope_files": len(files)}


@app.post("/embed")
async def embed(req: Request):
    """Atomic embedding primitive: embed caller-supplied texts and APPEND the vectors to a jsonl
    file in the model's working directory. Returns only metadata (path, count, dim) - NOT the
    vectors themselves, which would blow up the context. The model reads the jsonl back with its
    own tools (cat/python) and decides how to use the embeddings (similarity, clustering, etc).

    body: {texts: [str], role?: "query"|"passage" (default passage), out?: filename}
    Each jsonl line: {"i": <index in this call>, "text": <input text>, "role": <role>,
                      "dim": <D>, "embedding": [floats]}  (L2-normalized, so dot == cosine)
    return: {path, relpath, count, dim, role, appended} or {error}
    """
    body = await req.json()
    texts = body.get("texts")
    if isinstance(texts, str):
        texts = [texts]
    if not isinstance(texts, list) or not texts or not all(isinstance(t, str) for t in texts):
        return {"error": "texts must be a non-empty array of strings"}
    role = (body.get("role") or "passage").lower()
    if role not in ("query", "passage"):
        role = "passage"
    # Output file: model-chosen name (basename only, no traversal) else a stable default.
    out_name = os.path.basename(str(body.get("out") or "embeddings.jsonl")) or "embeddings.jsonl"
    out_path = os.path.join(WORK_DIR, out_name)

    embs = _encode(texts, role=role)
    dim = int(embs.shape[1]) if embs.ndim == 2 and embs.shape[0] else 0
    appended = 0
    with open(out_path, "a") as fh:
        for i, (t, v) in enumerate(zip(texts, embs)):
            fh.write(json.dumps({
                "i": i, "text": t, "role": role, "dim": dim,
                "embedding": [round(float(x), 6) for x in v.tolist()],
            }) + "\n")
            appended += 1
    try:
        rel = os.path.relpath(out_path, WORK_DIR)
    except Exception:
        rel = out_name
    return {"path": out_path, "relpath": rel, "count": appended, "dim": dim, "role": role,
            "appended": appended}


def _rerank_texts(q: str, docs: list, top_n=None) -> list:
    """Cross-encoder rerank helper -> [{index,score,text}] sorted desc. index is into docs."""
    docs = list(docs)
    if not q or not docs:
        return []
    if RERANK_BACKEND == "api":
        payload = {"model": API_RERANK_MODEL, "query": q, "documents": docs}
        if top_n is not None:
            payload["top_n"] = int(top_n)
        d = _api_post("rerank", payload)
        return [{"score": float(r.get("relevance_score", 0.0)),
                 "text": (r.get("document") or {}).get("text", "") if isinstance(r.get("document"), dict)
                         else r.get("document", ""),
                 "index": int(r.get("index", -1))} for r in d.get("results", [])]
    m = rerank_model()
    kwargs = {"top_n": int(top_n)} if top_n is not None else {}
    ranked = m.rerank(q, docs, **kwargs)
    return [{"score": float(r.get("relevance_score", 0.0)),
             "text": r.get("document", ""), "index": int(r.get("index", -1))} for r in ranked]


@app.post("/rerank")
async def rerank(req: Request):
    """Pure cross-encoder rerank with jina-reranker-v3: score caller-supplied documents.

    Deliberately does NOT fetch its own candidates (no hidden embedding search). The tool stays
    a single-model primitive (basic reranker usage) - the model itself decides what to feed it.
    """
    body = await req.json()
    q = body.get("query", "")
    docs = body.get("documents")
    top_n = body.get("top_n")
    if not q or not docs:
        return {"results": []}
    docs = list(docs)
    if RERANK_BACKEND == "api":
        payload = {"model": API_RERANK_MODEL, "query": q, "documents": docs}
        if top_n is not None:
            payload["top_n"] = int(top_n)
        d = _api_post("rerank", payload)
        out = [{"score": round(float(r.get("relevance_score", 0.0)), 4),
                "text": (r.get("document") or {}).get("text", "") if isinstance(r.get("document"), dict)
                        else r.get("document", ""),
                "index": int(r.get("index", -1))} for r in d.get("results", [])]
        return {"results": out}
    m = rerank_model()
    kwargs = {"top_n": int(top_n)} if top_n is not None else {}
    ranked = m.rerank(q, docs, **kwargs)
    out = [{"score": round(float(r.get("relevance_score", 0.0)), 4),
            "text": r.get("document", ""), "index": int(r.get("index", -1))} for r in ranked]
    return {"results": out}


# --- derived compute endpoints (all reduce to _encode / rerank; numpy on the server) ---------
# These back the many tool "shapes" the agent can pick from. Every one is just embedding or
# reranking under the hood - different I/O ergonomics, same two models.

def _embed_norm(texts, role: str) -> np.ndarray:
    e = _encode(list(texts), role=role)
    e = np.asarray(e, dtype=np.float32)
    n = np.linalg.norm(e, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return e / n


@app.post("/similarity")
async def similarity(req: Request):
    """Cosine similarity. Either pairwise (a,b same length) or matrix (queries x documents)."""
    body = await req.json()
    a = body.get("a") or body.get("queries") or []
    b = body.get("b") or body.get("documents") or []
    if not a or not b:
        return {"error": "need two non-empty text lists"}
    mode = body.get("mode") or ("pairwise" if len(a) == len(b) and not body.get("matrix") else "matrix")
    qa = _embed_norm(a, role="query")
    db = _embed_norm(b, role="passage")
    if mode == "pairwise" and len(a) == len(b):
        sims = [round(float(np.dot(qa[i], db[i])), 4) for i in range(len(a))]
        return {"mode": "pairwise", "similarities": sims}
    mat = np.round(qa @ db.T, 4).tolist()
    return {"mode": "matrix", "matrix": mat}


@app.post("/classify")
async def classify(req: Request):
    """Zero-shot classification via embeddings: assign each text its nearest label."""
    body = await req.json()
    texts = body.get("texts") or []
    labels = body.get("labels") or []
    if not texts or not labels:
        return {"error": "need texts and labels"}
    te = _embed_norm(texts, role="query")
    le = _embed_norm(labels, role="passage")
    sims = te @ le.T
    out = []
    for i, t in enumerate(texts):
        j = int(np.argmax(sims[i]))
        out.append({"text": t, "label": labels[j], "score": round(float(sims[i][j]), 4)})
    return {"results": out}


def _greedy_diverse(embs: np.ndarray, k: int) -> list:
    """Facility-location-style greedy: pick k items maximizing min-distance to chosen set."""
    n = embs.shape[0]
    k = max(1, min(k, n))
    chosen = [0]
    sims = embs @ embs.T
    while len(chosen) < k:
        # for each candidate, its max similarity to any chosen (lower = more diverse)
        maxsim = sims[:, chosen].max(axis=1)
        for c in chosen:
            maxsim[c] = 2.0
        chosen.append(int(np.argmin(maxsim)))
    return chosen


@app.post("/deduplicate")
async def deduplicate(req: Request):
    """Top-k semantically diverse subset of strings (submodular/greedy over embeddings)."""
    body = await req.json()
    strings = body.get("strings") or body.get("texts") or []
    if not strings:
        return {"error": "need strings"}
    k = int(body.get("k") or max(1, len(strings) // 2))
    embs = _embed_norm(strings, role="passage")
    idx = _greedy_diverse(embs, k)
    return {"kept_indices": idx, "kept": [strings[i] for i in idx]}


@app.post("/cluster")
async def cluster(req: Request):
    """Greedy threshold clustering over embeddings (no sklearn): group near-duplicate texts."""
    body = await req.json()
    texts = body.get("texts") or []
    if not texts:
        return {"error": "need texts"}
    thr = float(body.get("threshold", 0.82))
    embs = _embed_norm(texts, role="passage")
    sims = embs @ embs.T
    assigned = [-1] * len(texts)
    clusters = []
    for i in range(len(texts)):
        if assigned[i] != -1:
            continue
        cid = len(clusters)
        members = [i]
        assigned[i] = cid
        for j in range(i + 1, len(texts)):
            if assigned[j] == -1 and sims[i][j] >= thr:
                assigned[j] = cid
                members.append(j)
        clusters.append(members)
    return {"n_clusters": len(clusters),
            "clusters": [{"id": c, "members": m, "texts": [texts[x] for x in m]}
                        for c, m in enumerate(clusters)]}


@app.post("/stats")
async def stats(_: Request):
    """List the dataroom files. Does NOT embed anything - there is no boot index."""
    files = sorted(os.path.relpath(f, DATAROOM_DIR) for f in _dataroom_files())
    return {"files": files, "file_count": len(files),
            "embedded_scopes": len(_scope_cache),
            "embed_cache": dict(_embed_cache_stats, unique=len(_embed_cache)),
            "embed_backend": EMBED_BACKEND, "rerank_backend": RERANK_BACKEND,
            "embed_model": API_EMBED_MODEL if EMBED_BACKEND == "api" else EMBED_MODEL,
            "rerank_model": API_RERANK_MODEL if RERANK_BACKEND == "api" else RERANK_MODEL,
            "dataroom_dir": DATAROOM_DIR}


if __name__ == "__main__":
    # No index is built at boot. Models load lazily on first /search or /rerank.
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("DATAROOM_PORT", "8078")))
