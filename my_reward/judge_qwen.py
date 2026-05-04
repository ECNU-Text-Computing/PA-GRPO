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
"""Reward function for the pairwise Judge task with Qwen3 models.

Identical pairwise consistency logic to ``judge_llama.py`` (Eq. 8); differs in
the chain-of-thought tag (``<thinking>`` vs ``<think>``) and uses a richer
multi-strategy answer extractor with confidence scoring.
"""

import os
import re
import torch
from datetime import datetime
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter

# 获取当前文件所在目录，用于构建相对路径
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_CURRENT_DIR)

# 日志文件路径 (相对于项目根目录)
LOG_PATH = os.path.join(_PROJECT_ROOT, "logs", "judge_qwen3_8b.log")

# TensorBoard 日志目录
TB_LOG_DIR = os.path.join(_PROJECT_ROOT, "logs", "tb_judge_qwen3_8b")
_writer = None
_global_step = 0

# A<->B 的镜像映射（chatbot_arena只有A/B两个选项）
CONSIST_MAP = {
    "A": "B",
    "B": "A",
}

# 最大日志输出长度（避免过长文本刷屏）
MAX_OUTPUT_LOG_LEN = 2000


def extract_answer_with_confidence(text):
    """Extract the answer letter from a Qwen3 generation with a confidence score.

    Tries multiple strategies in order of decreasing confidence and returns the
    first match.

    Returns:
        Tuple ``(answer, confidence, method)``.

        Confidence levels:
            1.0 — standard ``<answer>X</answer>`` tag.
            0.8 — ``Final Answer: X`` form.
            0.6 — a standalone A/B at the tail of the text.
            0.5 — pattern match (e.g. "the answer is A").
            0.3 — last A/B character in the text (lowest-confidence fallback).
            0.0 — not found.
    """
    # 方法1: 查找 <answer> 标签（chatbot_arena标准格式，最高置信度）
    start = text.find("<answer>")
    if start != -1:
        end = text.find("</answer>", start)
        if end != -1:
            answer_text = text[start + 8:end].strip()
            if answer_text in ["A", "B"]:
                return answer_text, 1.0, "answer_tag"
    
    # 方法2: 查找 "Final Answer:" 格式
    start = text.find("Final Answer:")
    if start != -1:
        remaining = text[start + len("Final Answer:"):].lstrip()
        if remaining and remaining[0] in ["A", "B"]:
            return remaining[0], 0.8, "final_answer"
    
    tail = text[-100:] if len(text) > 100 else text
    match = re.search(r'(?:^|[\s\.:,;!\?])(A|B)(?:[\s\.:,;!\?]|$)', tail)
    if match:
        return match.group(1), 0.6, "tail_standalone"
    
    lower_text = text.lower()
    patterns = [
        r'(?:answer\s+is|choose|选择|选)\s*[:\s]?\s*(A|B)',
        r'(?:correct\s+answer\s+is|the\s+answer\s+is)\s*[:\s]?\s*(A|B)',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            ans = match.group(1).upper()
            if ans in ["A", "B"]:
                return ans, 0.5, "pattern_match"
    
    for char in reversed(text):
        if char in ["A", "B"]:
            return char, 0.3, "last_occurrence"
    
    return None, 0.0, "not_found"


def extract_answer(text):
    """Backwards-compatible wrapper that returns only the answer letter."""
    ans, _, _ = extract_answer_with_confidence(text)
    return ans


def extract_think_content(text):
    """Extract chain-of-thought content; tries ``<thinking>`` then ``<think>``."""
    # 优先查找 <thinking> 标签（数据集格式）
    start = text.find("<thinking>")
    if start != -1:
        end = text.find("</thinking>", start)
        if end != -1:
            return text[start + 10:end].strip() 
    
    # 兜底查找 <think> 标签（Qwen3原生格式）
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


def safe_log_output(index, pid, cuda, output_text):
    """Log the full model output, truncated to ``MAX_OUTPUT_LOG_LEN`` characters."""
    try:
        t = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        # 截断过长输出
        if len(output_text) > MAX_OUTPUT_LOG_LEN:
            truncated = output_text[:MAX_OUTPUT_LOG_LEN] + f"... [TRUNCATED, total {len(output_text)} chars]"
        else:
            truncated = output_text
        truncated = truncated.replace("\n", "\\n")
        line = f"{t} | idx={index} | pid={pid} | cuda={cuda} | OUTPUT | {truncated}\n"
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
    """Batch reward function for verl's ``BatchRewardManager`` (Judge / Qwen3).

    See ``judge_llama.compute_score`` for the index-aligned pairwise consistency
    rule (Eq. 8). The Qwen3 variant additionally records a per-sample answer-
    extraction confidence via ``extract_answer_with_confidence``.

    Args:
        data_sources: List[str], e.g. ``"chatbot_arena"``.
        solution_strs: List[str], model-generated outputs.
        ground_truths: List[str], gold answers (A or B).
        extra_infos: List[dict], each entry must contain ``index`` and
            ``permutation`` (0 or 1) fields.
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
    confidences = [0.0] * N
    extraction_methods = [None] * N
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
    confidence_list = []

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

        safe_log_output(idx, pid, cuda_device, sol)

        ans, confidence, method = extract_answer_with_confidence(sol)
        answers[i] = ans
        confidences[i] = confidence
        extraction_methods[i] = method
        confidence_list.append(confidence)

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

        # 收集各分量
        corr_list.append(score_corr)
        format_list.append(bonus_format)
        length_list.append(length_reward)

        # ==== log 基础信息（含置信度和提取方法） ====
        safe_log(
            idx,
            pid,
            cuda_device,
            (
                f"BASE | i={i} | pair_id={pair_id} | perm={perm} | "
                f"ans={ans} | gt={gt} | confidence={confidence:.2f} | method={method} | "
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
                        f"i0={i0},idx0={idx0},ans0={a0},conf0={confidences[i0]:.2f} | "
                        f"i1={i1},idx1={idx1},ans1={a1},conf1={confidences[i1]:.2f} | "
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

            # ==== log一致性奖励 ====
            safe_log(
                f"{idx0},{idx1}",
                pid,
                cuda_device,
                (
                    "PAIR_CHECK | "
                    f"pair_id={pid_} | t={t} | "
                    f"i0={i0},idx0={idx0},ans0={a0},conf0={confidences[i0]:.2f} | "
                    f"i1={i1},idx1={idx1},ans1={a1},conf1={confidences[i1]:.2f} | "
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
    mean_confidence = safe_mean(confidence_list)
    mean_total = safe_mean(scores)

    writer.add_scalar("reward/correctness", mean_corr, _global_step)
    writer.add_scalar("reward/format", mean_format, _global_step)
    writer.add_scalar("reward/length", mean_length, _global_step)
    writer.add_scalar("reward/consistency", mean_consistency, _global_step)
    writer.add_scalar("reward/confidence", mean_confidence, _global_step)
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
        "confidence": [float(x) for x in confidences],
        "extraction_method": extraction_methods,
        "total": [float(x) for x in scores],
    }
    return {
        "reward_tensor": rewards_tensor,
        "reward_extra_info": reward_extra_info,
    }
