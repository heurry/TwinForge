# Reuse the project's local offline image: it already contains PyTorch,
# Transformers, FastAPI and Uvicorn. The runtime is CPU-only by default so it
# cannot contend with the two-GPU chat serving workload.
ARG EMBEDDING_BASE_IMAGE=local/train:qwen35-v1
FROM ${EMBEDDING_BASE_IMAGE}

WORKDIR /app
COPY app.py /app/app.py

ENV EMBEDDING_MODEL_PATH=/models/Qwen3-Embedding-0.6B \
    EMBEDDING_MODEL_NAME=qwen3-embedding-0.6b \
    EMBEDDING_MAX_LENGTH=1024 \
    EMBEDDING_MAX_BATCH_SIZE=16 \
    EMBEDDING_CPU_THREADS=8 \
    EMBEDDING_DTYPE=float32 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_OFFLINE=1

EXPOSE 8000
ENTRYPOINT []
CMD ["python3", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
