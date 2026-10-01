#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
model_source="${TRAINING_MODEL_SOURCE:-/mnt/nvme-data/models/LLM_model/Qwen3.5-4B}"
dataset_source="${TRAINING_DATASET_SOURCE:-${root_dir}/data/cleaned/dianjin_csc_sft_train.jsonl}"
node_root="/opt/twinforge"

if [[ ! -f "${dataset_source}" ]]; then
  python_bin="${root_dir}/.venv/bin/python"
  [[ -x "${python_bin}" ]] || python_bin=python3
  "${python_bin}" "${root_dir}/scripts/prepare_dianjin_sft.py"
fi

if ! minikube status >/dev/null 2>&1; then
  echo "Minikube is not running. Start it first with: minikube start" >&2
  exit 1
fi

docker run --rm --privileged alpine:3.20 \
  sysctl -w fs.inotify.max_user_instances=1024 fs.inotify.max_user_watches=524288 >/dev/null

api_port="$(docker port minikube 8443/tcp | head -n 1 | awk -F: '{print $NF}')"
if [[ -z "${api_port}" ]]; then
  echo "Unable to resolve the Minikube API host port" >&2
  exit 1
fi
kubectl_local=(kubectl "--server=https://127.0.0.1:${api_port}")

manifest="$(mktemp)"
trap 'rm -f "${manifest}"' EXIT
kubectl kustomize "${root_dir}/deploy/kubeflow" >"${manifest}"
"${kubectl_local[@]}" apply --server-side -f "${manifest}"
"${kubectl_local[@]}" -n kubeflow rollout status deployment/training-operator --timeout=180s

docker exec minikube mkdir -p \
  "${node_root}/models" "${node_root}/data" "${node_root}/artifacts/training" /root/training-build
docker exec minikube rm -rf "${node_root}/models/Qwen3.5-4B"
docker cp "${model_source}" minikube:"${node_root}/models/Qwen3.5-4B"
docker cp "${dataset_source}" minikube:"${node_root}/data/dianjin_csc_sft_train.jsonl"
docker cp "${root_dir}/deploy/training/Dockerfile" minikube:/root/training-build/Dockerfile
docker cp "${root_dir}/deploy/training/entrypoint.sh" minikube:/root/training-build/entrypoint.sh
docker cp "${root_dir}/deploy/training/train_sft.py" minikube:/root/training-build/train_sft.py
docker exec minikube sh -c 'cd /root/training-build && docker build -t local/train:qwen35-v1 .'

"${kubectl_local[@]}" -n training create secret generic training-artifacts \
  --from-literal=S3_ENDPOINT=http://192.168.49.1:9000 \
  --from-literal=S3_ACCESS_KEY=minioadmin \
  --from-literal=S3_SECRET_KEY=minioadmin \
  --from-literal=S3_BUCKET=infra-artifacts \
  --dry-run=client -o yaml | "${kubectl_local[@]}" apply -f -

echo "Training runtime ready: PyTorchJob CRD, 2-GPU node assets, local/train:qwen35-v1, MinIO secret"
