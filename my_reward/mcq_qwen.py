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
"""Reward function for the MCQ task with Qwen3 models.

Identical in structure to ``mcq_llama.py`` but parses the ``<thinking>`` tag
used by Qwen3's chat template instead of ``<think>``. Implements Eq. 7, 9, 10
of the paper.
"""

import os
import torch
from datetime import datetime
from collections import defaultdict, Counter
from torch.utils.tensorboard import SummaryWriter

# 获取当前文件所在目录，用于构建相对路径
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_CURRENT_DIR)

# 日志文件路径 (相对于项目根目录)
LOG_PATH = os.path.join(_PROJECT_ROOT, "logs", "mcq_qwen3_8b.log")

# TensorBoard 日志目录
TB_LOG_DIR = os.path.join(_PROJECT_ROOT, "logs", "tb_mcq_qwen3_8b")
_writer = None
_global_step = 0


def map_to_canonical(ans, permutation):
    """Map a permuted answer letter back to the canonical (original) option.

    Implements the label-to-semantic mapping m_τ described in Section 3.3.1
    of the paper. See ``mcq_llama.map_to_canonical`` for a worked example.

    Args:
        ans: Model output letter (A/B/C/D).
        permutation: 4-character permutation string (e.g. "BCDA", "DCBA").

    Returns:
        The canonical letter (A/B/C/D), or ``None`` if the input is invalid.
    """
    CANONICAL_CHOICES = "ABCD"
    if not ans:
        return None
    ans = ans.upper()
    if ans not in CANONICAL_CHOICES:
        return None
    
    if not permutation or len(permutation) != 4:
        return ans
    
    try:
        idx = CANONICAL_CHOICES.index(ans)
        return permutation[idx]
    except (IndexError, ValueError):
        return None

def extract_answer(text):
    """Extract the answer letter from an ``<answer>X</answer>`` tag, X ∈ {A,B,C,D}."""
    import re
    # 查找 <answer>...</answer> 标签
    match = re.search(r'<answer>\s*([A-D])\s*</answer>', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return None

def extract_think_content(text):
    """Extract the chain-of-thought content inside ``<thinking>...</thinking>`` tags."""
    import re
    match = re.search(r'<thinking>(.*?)</thinking>', text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
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
    """Batch reward function for verl's ``BatchRewardManager`` (Qwen3 variant).

    See ``mcq_llama.compute_score`` for the full description of the cross-
    permutation unique-mode consistency rule (Eq. 9, 10) and the preliminary
    reward (Eq. 7). The only behavioral difference from the Llama variant is
    the chain-of-thought tag (``<thinking>`` vs ``<think>``).

    Args:
        data_sources: List[str], typically unused.
        solution_strs: List[str], model-generated outputs.
        ground_truths: List[str] or ``None``, gold answers in canonical ABCD form.
        extra_infos: List[dict], each entry must contain ``original_index`` and
            ``permutation`` fields.
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
    canonical_answers = [None] * N  # 映射回原始选项的答案
    original_indices = [None] * N
    permutations = [None] * N
    correctness_scores = [0.0] * N
    format_scores = [0.0] * N
    length_scores = [0.0] * N
    consistency_scores = [0.0] * N

    # 添加分量列表用于统计
    corr_list = []
    format_list = []
    length_list = []
    consistency_list = []

    # ========= 1. 先算基础 reward  =========
    for i in range(N):
        extra = extra_infos[i] or {}

        # 从 extra_info 读取关键字段
        original_idx = extra.get("original_index", i)
        permutation = extra.get("permutation", "ABCD")
        
        original_indices[i] = original_idx
        permutations[i] = permutation

        sol = solution_strs[i]
        gt = ground_truths[i]

        ans = extract_answer(sol)
        answers[i] = ans
        
        # 将答案映射回原始选项（canonical answer）- 使用通用映射函数
        canonical_ans = map_to_canonical(ans, permutation)
        canonical_answers[i] = canonical_ans

        # ==== 原始打分逻辑 ====
        if gt is None:
            base_score = 0.0
            score_corr = 0.0
            bonus_format = 0.0
            length_reward = 0.0
        else:
            # 正确性判断：使用 canonical_ans 与 gt 比较（gt 应该是原始选项）
            # 如果无法解析出 canonical_ans（格式错或 permutation 错），算作错误
            if canonical_ans is None:
                score_corr = -1.0
            else:
                score_corr = 1.0 if canonical_ans == gt else -1.0

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

        # 收集各分量
        corr_list.append(score_corr)
        format_list.append(bonus_format)
        length_list.append(length_reward)

        # ==== log 基础信息 ====
        safe_log(
            original_idx,
            pid,
            cuda_device,
            (
                f"BASE | i={i} | original_idx={original_idx} | perm={permutation} | "
                f"ans={ans} | canonical_ans={canonical_ans} | gt={gt} | "
                f"base_score={base_score}"
            ),
        )

    # ========= 2. 跨 permutation 按对应位置配对，计算一致性奖励 =========
    # group[original_idx][permutation] = [本 batch 中所有出现位置 i，按出现顺序]
    group = defaultdict(lambda: defaultdict(list))
    for i in range(N):
        orig_idx = original_indices[i]
        perm = permutations[i]
        if orig_idx is not None and perm is not None:
            group[orig_idx][perm].append(i)

    # 对每个 original_idx，跨 permutation 按对应位置配对
    for orig_idx, perm_dict in group.items():
        all_perms = list(perm_dict.keys())
        if len(all_perms) <= 1:
            continue
        
        min_rollouts = min(len(perm_dict[p]) for p in all_perms)
        if min_rollouts == 0:
            continue
        
        for t in range(min_rollouts):
            indices_at_t = [perm_dict[p][t] for p in all_perms]
            
            canonical_answers_at_t = [canonical_answers[i] for i in indices_at_t]
            valid_answers = [ans for ans in canonical_answers_at_t if ans is not None]
            
            if len(valid_answers) == 0:
                safe_log(
                    orig_idx,
                    pid,
                    cuda_device,
                    f"CROSS_PERM_SKIP | original_idx={orig_idx} | t={t} | reason=no_valid_answers | indices={indices_at_t}",
                )
                continue
            
            # 计算众数
            answer_counter = Counter(valid_answers)
            consensus_ans, max_count = answer_counter.most_common(1)[0]
            
            has_strong_consensus = (max_count >= 3)
            
            if not has_strong_consensus:
                safe_log(
                    orig_idx,
                    pid,
                    cuda_device,
                    f"CROSS_PERM_WEAK_CONSENSUS | original_idx={orig_idx} | t={t} | max_count={max_count} < 3 | giving negative reward to all | answers={valid_answers}",
                )
                for i in indices_at_t:
                    consistency_bonus = -consistency_weight
                    scores[i] += consistency_bonus
                    consistency_scores[i] += consistency_bonus
                    consistency_list.append(consistency_bonus)
                continue
            
            
            for i in indices_at_t:
                canonical_ans = canonical_answers[i]
                
                if canonical_ans is None:
                    consistency_bonus = -consistency_weight
                elif canonical_ans == consensus_ans:
                    consistency_bonus = consistency_weight
                else:
                    consistency_bonus = -consistency_weight
                
                scores[i] += consistency_bonus
                consistency_scores[i] += consistency_bonus
                consistency_list.append(consistency_bonus)
                                
            # ==== log 该位置的一致性信息 ====
            safe_log(
                orig_idx,
                pid,
                cuda_device,
                (
                    f"CROSS_PERM_CONSENSUS | original_idx={orig_idx} | t={t} | "
                    f"num_perms={len(all_perms)} | "
                    f"consensus_ans={consensus_ans} | max_count={max_count} | "
                    f"canonical_answers={canonical_answers_at_t} | "
                    f"indices={indices_at_t}"
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
    mean_consistency = safe_mean(consistency_list) if consistency_list else 0.0
    mean_total = safe_mean(scores)

    writer.add_scalar("reward/correctness", mean_corr, _global_step)
    writer.add_scalar("reward/format", mean_format, _global_step)
    writer.add_scalar("reward/length", mean_length, _global_step)
    writer.add_scalar("reward/consistency", mean_consistency, _global_step)
    writer.add_scalar("reward/total", mean_total, _global_step)

    _global_step += 1

    # ==== 最终汇总 log ====
    safe_log(
        "BATCH_SUMMARY",
        pid,
        cuda_device,
        (
            f"N={N} | "
            f"mean_corr={mean_corr:.3f} | "
            f"mean_format={mean_format:.3f} | "
            f"mean_length={mean_length:.3f} | "
            f"mean_consistency={mean_consistency:.3f} | "
            f"mean_total={mean_total:.3f}"
        ),
    )

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
