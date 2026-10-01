FROM local/vllm-openai:qwen35-v0.19.1

ARG PIP_INDEX_URL=https://pypi.org/simple

RUN python3 -m pip install --no-cache-dir \
    peft==0.19.1 \
    deepspeed==0.18.9

WORKDIR /workspace
COPY train_sft.py entrypoint.sh /workspace/
RUN chmod +x /workspace/entrypoint.sh

ENV PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

ENTRYPOINT ["/workspace/entrypoint.sh"]
