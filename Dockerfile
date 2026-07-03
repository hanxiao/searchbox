# Searchbox app: reuse the dataroom image (torch + node + cached local retrieval stack), bump pi
# to the version searchbox pins, overlay searchbox source. CPU embedder; LLM via shared llama.
FROM ghcr.io/hanxiao/dataroom:latest

# Link the published image to the repo on GHCR (Packages -> repository).
LABEL org.opencontainers.image.source="https://github.com/hanxiao/searchbox"
LABEL org.opencontainers.image.description="Searchbox: budget-bounded closed-dataroom QA harness"
LABEL org.opencontainers.image.licenses="MIT"

ENV PI_BIN=pi PI_SKIP_VERSION_CHECK=1
ARG PI_VERSION=0.80.1
RUN npm install -g @earendil-works/pi-coding-agent@${PI_VERSION} && npm cache clean --force
# Patch pi auto-compaction re-entrancy race (Cannot read properties of undefined (reading signal)).
# Idempotent; see scripts/pi_compaction_patch.py. Re-applied on every rebuild so a pi bump keeps it.
COPY scripts/pi_compaction_patch.py /tmp/pi_compaction_patch.py
RUN python3 /tmp/pi_compaction_patch.py "$(npm root -g)/@earendil-works/pi-coding-agent/dist/core/agent-session.js" && node --check "$(npm root -g)/@earendil-works/pi-coding-agent/dist/core/agent-session.js"

WORKDIR /app
# searchbox-specific python deps (transformers/sentence-transformers pins). torch already present.
COPY server/requirements.txt /app/sb-requirements.txt
RUN pip install -r /app/sb-requirements.txt

# Overlay searchbox source (replaces dataroom's server/web/pi with searchbox's)
COPY server /app/server
COPY web /app/web
COPY pi /app/pi
COPY config /app/config
COPY scripts /app/scripts
COPY PI_VERSION /app/PI_VERSION

ENV JOBS_DIR=/data/jobs \
    EMBED_BACKEND=local \
    RERANK_BACKEND=local \
    EMBED_MODEL=jinaai/jina-embeddings-v5-text-nano \
    RERANK_MODEL=jinaai/jina-reranker-v2-base-multilingual \
    MODEL_ID=qwen3.6 \
    CONTEXT_WINDOW=131072 \
    PORT=8001
EXPOSE 8001
CMD ["python","-m","server.app"]
