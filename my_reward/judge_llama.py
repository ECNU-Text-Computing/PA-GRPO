# Copyright 2026 East China Normal University
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Reward function for the pairwise Judge task with Llama models.

Implements the index-aligned pairwise consistency reward for P = 2 (Eq. 8 of
the paper). Used by `scripts/run_judge_llama.sh`.
"""

import os
import torch
from datetime import datetime
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter

# 获取当前文件所在目录，用于构建相对路径
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_CURRENT_DIR)

# 日志文件路径 (相对于项目根目录)
LOG_PATH = os.path.join(_PROJECT_ROOT, "logs", "judge_llama3_1_8b.log")

# TensorBoard 日志目录
TB_LOG_DIR = os.path.join(_PROJECT_ROOT, "logs", "tb_judge_llama3_1_8b")
_writer = None
_global_step = 0

CONSIST_MAP = {
    "A": "B",
    "B": "A",
}

def extract_answer(text):
    """Extract the answer letter X ∈ {A, B}.

    First tries ``<answer>X</answer>``; falls back to ``Final Answer: X``.
    """
    start = text.find("<answer>")
    if start != -1:
        end = text.find("</answer>", start)
        if end != -1:
            answer_text = text[start + 8:end].strip()
            if answer_text in ["A", "B"]:
                return answer_text
    
    start = text.find("Final Answer:")
    if start != -1:
        remaining = text[start + len("Final Answer:"):].lstrip()
        if remaining and remaining[0] in ["A", "B"]:
            return remaining[0]
    
    return None

def extract_think_content(text):
    """Extract the chain-of-thought content inside ``<think>...</think>`` tags."""
    start = text.find("<think>")
    if start != -1:
        end = text.find("</think>", start)
        if end != -1:
            return text[start + 7:end].strip()
    return None

def safe_log(index, pid, cuda, txt):
    """Append a single log line to ``LOG_PATH`` without ``logging`` contention.

    Args:
        index: Sample identifier; may be an ``int`` or a string like ``"idx0,idx1"``.
        pid: Process ID of the caller.
        cuda: CUDA device string for the caller.
        txt: Pre-formatted message body.
    """
    try:
        t = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        line = f"{t} | idx={index} | pid={pid} | cuda={cuda} | {txt}\n"
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception:
        pass

def get_tb_writer():
    """Return the lazily-initialized global TensorBoard writer."""
    global _writer
    if _writer is None:
        os.makedirs(TB_LOG_DIR, exist_ok=True)
        _writer = SummaryWriter(log_dir=TB_LOG_DIR)
    return _writer

def compute_score(
    data_sources,
    solution_strs,
    ground_truths,
    extra_infos=None,
    consistency_weight: float = 1,
    return_dict: bool = False,
):
    """Batch reward function for verl's ``BatchRewardManager`` (Judge / Llama).

    For the pairwise Judge task with permutation set Π = {AB, BA}, applies the
    index-aligned pairwise consistency rule of Eq. 8: pairs the t-th rollout
    under permutation 0 with the t-th rollout under permutation 1; the pair is
    consistent iff the two answers are mirrors (A ↔ B). Consistent pairs receive
    +``consistency_weight``, inconsistent pairs receive -``consistency_weight``.

    Args:
        data_sources: List[str], data source identifiers.
        solution_strs: List[str], model-generated outputs.
        ground_truths: List[str], gold answers (A or B).
        extra_infos: List[dict], each entry must contain ``index`` and
            ``permutation`` (0 or 1) fields populated by the data preprocessor.
        consistency_weight: Coefficient λ for the consistency reward (Eq. 6).
        return_dict: If True, return a dict with both the reward tensor and
            per-component breakdowns; otherwise return only the tensor.

    Returns:
        ``torch.FloatTensor`` of shape ``(N,)`` (or a dict, see ``return_dict``).
    """
    if extra_infos is None:
        extra_infos = [dict() for _ in range(len(solution_strs))]

    pid = os.getpid()
    cuda_device = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "CPU"
    N = len(solution_strs)
    scores = [0.0] * N
    answers = [None] * N
    pair_ids = [None] * N
    permutations = [None] * N
    idx_list = [None] * N
    correctness_scores = [0.0] * N
    format_scores = [0.0] * N
    length_scores = [0.0] * N
    consistency_scores = [0.0] * N

    # 添加分量列表用于统计
    corr_list = []
    format_list = []
    length_list = []

    # ========= 1. 先算基础 reward =========
    for i in range(N):
        extra = extra_infos[i] or {}

        idx = extra.get("index", i)
        idx_list[i] = idx

        try:
            idx_int = int(idx)
        except Exception:
            idx_int = i 

        perm = extra.get("permutation", idx_int % 2)
        permutations[i] = perm
        
        pair_id = idx_int // 2
        pair_ids[i] = pair_id

        sol = solution_strs[i]
        gt = ground_truths[i]

        ans = extract_answer(sol)
        answers[i] = ans

        # ==== 打分逻辑 ====
        if gt is None:
            base_score = 0.0
            score_corr = 0.0
            bonus_format = 0.0
            length_reward = 0.0
        else:
            score_corr = 1.0 if ans == gt else -1.0

            bonus_format = 0.3 if "<answer>" in sol and "</answer>" in sol else -0.3

            think_content = extract_think_content(sol)
            if think_content is not None:
                think_len = len(think_content)
                if think_len < 500:
                    length_reward = -0.1     
                elif think_len < 2000:
                    length_reward = 0.1     
                elif think_len <= 3500:
                    length_reward = 0.2      
                elif think_len <= 5000:
                    length_reward = 0.1      
                else:
                    length_reward = -0.1     
            else:
                length_reward = -0.1

            base_score = score_corr + bonus_format + length_reward

        scores[i] = float(base_score)
        correctness_scores[i] = float(score_corr)
        format_scores[i] = float(bonus_format)
        length_scores[i] = float(length_reward)

        corr_list.append(score_corr)
        format_list.append(bonus_format)
        length_list.append(length_reward)

        # ==== log 基础信息 ====
        safe_log(
            idx,
            pid,
            cuda_device,
            (
                f"BASE | i={i} | pair_id={pair_id} | perm={perm} | "
                f"ans={ans} | gt={gt} | "
                f"base_score={base_score:.3f} | "
                f"corr={score_corr:.3f} | fmt={bonus_format:.3f} | len={length_reward:.3f}"
            ),
        )

    # ========= 2. 在 batch 内做成对一致性奖励 =========
    # group[ pair_id ][ permutation ] = [本 batch 中所有出现位置 i，按出现顺序]
    group = defaultdict(lambda: {0: [], 1: []})
    for i, (pid_, perm_) in enumerate(zip(pair_ids, permutations)):
        if pid_ is None:
            continue
        if perm_ not in (0, 1):
            continue
        group[pid_][perm_].append(i)


    for pid_, perm_dict in group.items():
        idxs0 = perm_dict[0]
        idxs1 = perm_dict[1]

        m = min(len(idxs0), len(idxs1))
        if m == 0:
            continue

        for t in range(m):
            i0 = idxs0[t]
            i1 = idxs1[t]

            idx0 = idx_list[i0]
            idx1 = idx_list[i1]

            a0 = answers[i0]
            a1 = answers[i1]

            if a0 not in CONSIST_MAP or a1 is None:
                safe_log(
                    f"{idx0},{idx1}",
                    pid,
                    cuda_device,
                    (
                        f"PAIR_SKIP | pair_id={pid_} | t={t} | "
                        f"i0={i0},idx0={idx0},ans0={a0} | "
                        f"i1={i1},idx1={idx1},ans1={a1} | "
                        "reason=invalid_answer_for_mapping"
                    ),
                )
                continue


            mapped = CONSIST_MAP.get(a0)
            is_consistent = (mapped == a1)

            if is_consistent:
                pair_bonus = consistency_weight
                scores[i0] += pair_bonus
                scores[i1] += pair_bonus
            else:
                pair_bonus = -consistency_weight
                scores[i0] += pair_bonus
                scores[i1] += pair_bonus
            consistency_scores[i0] += pair_bonus
            consistency_scores[i1] += pair_bonus

            # ==== log 这一对的一致性奖励 ====
            safe_log(
                f"{idx0},{idx1}",
                pid,
                cuda_device,
                (
                    "PAIR_CHECK | "
                    f"pair_id={pid_} | t={t} | "
                    f"i0={i0},idx0={idx0},ans0={a0} | "
                    f"i1={i1},idx1={idx1},ans1={a1} | "
                    f"mapped(ans0)={mapped} | "
                    f"is_consistent={is_consistent} | "
                    f"pair_bonus={pair_bonus:.3f} | "
                    f"final_score0={scores[i0]:.3f} | "
                    f"final_score1={scores[i1]:.3f}"
                ),
            )

    # ========= 3. 记录到 TensorBoard =========
    global _global_step
    writer = get_tb_writer()

    def safe_mean(xs):
        return float(sum(xs) / max(len(xs), 1)) if xs else 0.0

    mean_corr = safe_mean(corr_list)
    mean_format = safe_mean(format_list)
    mean_length = safe_mean(length_list)
    mean_consistency = safe_mean(consistency_scores)
    mean_total = safe_mean(scores)

    writer.add_scalar("reward/correctness", mean_corr, _global_step)
    writer.add_scalar("reward/format", mean_format, _global_step)
    writer.add_scalar("reward/length", mean_length, _global_step)
    writer.add_scalar("reward/consistency", mean_consistency, _global_step)
    writer.add_scalar("reward/total", mean_total, _global_step)

    _global_step += 1

    rewards_tensor = torch.tensor(scores, dtype=torch.float32)
    if not return_dict:
        return rewards_tensor
    reward_extra_info = {
        "correctness": [float(x) for x in correctness_scores],
        "format": [float(x) for x in format_scores],
        "length": [float(x) for x in length_scores],
        "consistency": [float(x) for x in consistency_scores],
        "total": [float(x) for x in scores],
    }
    return {
        "reward_tensor": rewards_tensor,
        "reward_extra_info": reward_extra_info,
    }
