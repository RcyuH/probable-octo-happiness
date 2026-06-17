# Reproduce Guidelines

## Environment Setup

- CUDA version: 12.6

- `pip install  --no-cache-dir torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126 --extra-index-url https://pypi.org/simple`

- `pip install sglang[all]>=0.4.10.post2`

- `pip3 install --no-cache-dir vllm==0.10.0`

- `pip3 install --no-cache-dir "transformers[hf_xet]>=4.51.0" hydra-core tensordict numpy pytest pybind11 codetiming`

- `pip3 install --no-cache-dir --use-pep517 flash-attn --no-build-isolation`

- instal apex following https://github.com/NVIDIA/apex

- `pip install -r requirements.txt`

## Environment Variables

Before running the scripts, set the following environment variables:
- `export MY_DATA_DIR=...`, Directory to store/download datasets.
- `export MY_MODEL_DIR=...`, Directory to store/download models.
  - you should download Qwen3/8B-base under the directory `${MY_MODEL_DIR}/Qwen/Qwen3-8B-Base`
- `export MY_CKPT_DIR=...`, Directory to store/download checkpoints.

## Run scripts

- initialize ray cluster `ray start --head`

- run the example script`bash run_grpo_pros.sh`
  > you may need to modify the script to fit your hardware

## Notes on Codebase

- This codebase is built on top of verl(https://github.com/volcengine/verl).

- The core algorithmic logic of our method (PROS) is implemented in `verl/experimental/dataset/tree_engine.py`

  - `update_data_source()` describes augmented query construction.

  - `update_posterior()` describes augmented query selection.

- **⚠️ If you have any issues about the code, please feel free to raise them in the review. We will address them during the rebuttal stage.**