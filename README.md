# PA-GRPO: Permutation-Aware Group Relative Policy Optimization

PA-GRPO mitigates **selection bias** in LLMs by jointly modeling permutation
variants of the same instance as a Permutation Group, encouraging
permutation-invariant decisions on MCQ and pairwise LLM-as-a-Judge tasks.

Forked from [verl](https://github.com/volcengine/verl) @ [`0eb50ec4`](https://github.com/volcengine/verl/commit/0eb50ec4a33cda97e05ed8caab9c7f17a30c05a9) (v0.7.0.dev).
PA-GRPO's core source-level change is concentrated in `verl/trainer/ppo/ray_trainer.py`: samples generated from different permutations of the same original question are grouped into the same GRPO group, and the within-group reward mean and variance are used to compute the permutation-aware advantage.

## Install (Python 3.12, CUDA 12.4)

```bash
git clone https://github.com/ECNU-Text-Computing/PA-GRPO.git && cd PA-GRPO
conda create -n pagrpo python=3.12 -y && conda activate pagrpo
pip install -r requirements-lock.txt   # exact paper environment
```
> Do **not** install `verl` separately; PA-GRPO uses the modified local `./verl/`.

## Run

```bash
# Train — set MODEL_PATH inside the script first
bash scripts/run_mcq_llama.sh   # or run_{mcq,judge}_{llama,qwen}.sh

# Inference (greedy; auto-discovers $PAGRPO_DATASET_DIR/*_<mode>.parquet)
python evaluation/evaluate_models.py --model_path /path/to/ckpt \
    --base_model_path /path/to/base --mode think --batch_size 32

# Score
python evaluation/compute_metrics_judge.py ./eval_results/<run>/   # P=2
python evaluation/compute_metrics_mcq.py   ./eval_results/<run>/   # P=24
```
Pre-built training parquets are included in `dataset/train/`. Place benchmark test parquets under `dataset/test/` with the `*_<mode>.parquet` naming pattern.

## Citation & License

```bibtex
@article{zheng2026mitigating,
  title={Mitigating Selection Bias in Large Language Models via Permutation-Aware GRPO},
  author={Zheng, Jinquan and Yuan, Jia and Yao, Jiacheng and Gu, Chenyang and Zheng, Pujun and He, Guoxiu},
  journal={arXiv preprint arXiv:2603.21016},
  year={2026},
  note={To appear in ACL 2026}
}
```
Apache 2.0. See [`LICENSE`](./LICENSE) and [`Notice.txt`](./Notice.txt).
