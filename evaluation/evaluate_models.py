#!/usr/bin/env python3
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
"""Inference driver for PA-GRPO. Runs a HF / LoRA / merged checkpoint over the
test parquets and writes per-sample JSON for ``compute_metrics_*.py`` to score.
Backend is vLLM by default (``--use_transformers`` for the slower fallback).
Test data is read from ``$PAGRPO_DATASET_DIR`` or ``<repo_root>/dataset/test/``.
"""

import os
import re
import glob
import json
import argparse
import torch
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
from pathlib import Path


# ==================== JSON序列化修复 ====================
def convert_to_serializable(obj):
    """将numpy类型转换为Python原生类型，确保JSON可序列化"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif pd.isna(obj):
        return None
    return obj


# ==================== 数据集配置 ====================
# Default location is `<repo_root>/dataset/test/` relative to this file. Override
# with the PAGRPO_DATASET_DIR environment variable to point elsewhere. Files are
# auto-discovered: `*_<mode>.parquet` under DATASET_DIR is picked up at runtime.
_DEFAULT_DATASET_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset", "test"
)
DATASET_DIR = os.environ.get("PAGRPO_DATASET_DIR", _DEFAULT_DATASET_DIR)


# ==================== 工具函数 ====================
def detect_model_type(model_path):
    """自动检测模型类型"""
    model_path_lower = model_path.lower()
    if "qwen" in model_path_lower:
        return "qwen3"
    return "llama"


def detect_checkpoint_type(checkpoint_path):
    """
    检测checkpoint类型
    
    Returns:
        dict: {
            "type": "lora" | "full_hf" | "base" | "verl_fsdp_unconverted",
            "lora_path": str or None,
            "hf_path": str or None,
            "base_model_hint": str or None,
            "needs_conversion": bool
        }
    """
    # 首先检查是否是 verl 的 FSDP checkpoint (未转换格式)
    # 特征：有 model_world_size_*_rank_*.pt 文件和 fsdp_config.json
    fsdp_config_path = os.path.join(checkpoint_path, "fsdp_config.json")
    model_pt_files = [f for f in os.listdir(checkpoint_path) if f.startswith("model_world_size_") and f.endswith(".pt")] if os.path.exists(checkpoint_path) else []

    if os.path.exists(fsdp_config_path) and len(model_pt_files) > 0:
        # 这是 verl 的 FSDP checkpoint，需要先转换为 HuggingFace 格式
        hf_subdir = os.path.join(checkpoint_path, "huggingface")
        print(f"\n{'='*70}")
        print(f"WARNING: detected an unconverted verl FSDP checkpoint")
        print(f"path: {checkpoint_path}")
        print(f"shard files: {model_pt_files}")
        print(f"\nThis checkpoint must be merged into HuggingFace format before it")
        print(f"can be loaded. From the project root, run:")
        print(f"\n  python -m verl.model_merger merge \\")
        print(f"      --backend fsdp \\")
        print(f"      --local_dir {checkpoint_path} \\")
        print(f"      --target_dir {checkpoint_path}_merged")
        print(f"\nThen pass {checkpoint_path}_merged as --model_path.")
        print(f"{'='*70}\n")
        return {
            "type": "verl_fsdp_unconverted",
            "lora_path": None,
            "hf_path": None,
            "base_model_hint": None,
            "needs_conversion": True
        }
    
    actor_path = os.path.join(checkpoint_path, "actor")
    lora_path = os.path.join(actor_path, "lora_adapter")
    hf_path = os.path.join(actor_path, "huggingface")
    
    result = {
        "type": "base",
        "lora_path": None,
        "hf_path": None,
        "base_model_hint": None,
        "needs_conversion": False
    }
    
    # 检查是否是checkpoint目录
    if not os.path.exists(actor_path):
        # 也检查根目录下是否有 lora_adapter（转换后的格式）
        root_lora_path = os.path.join(checkpoint_path, "lora_adapter")
        if os.path.exists(root_lora_path) and os.path.exists(os.path.join(root_lora_path, "adapter_config.json")):
            result["type"] = "lora"
            result["lora_path"] = root_lora_path
            try:
                with open(os.path.join(root_lora_path, "adapter_config.json"), 'r') as f:
                    config = json.load(f)
                    result["base_model_hint"] = config.get("base_model_name_or_path")
            except:
                pass
            # 检查是否有完整模型权重
            has_weights = any(f.endswith(('.safetensors', '.bin')) for f in os.listdir(checkpoint_path) if os.path.isfile(os.path.join(checkpoint_path, f)))
            if has_weights:
                result["hf_path"] = checkpoint_path
        # 检查根目录是否直接有模型权重
        elif os.path.exists(checkpoint_path):
            has_weights = any(f.endswith(('.safetensors', '.bin')) for f in os.listdir(checkpoint_path) if os.path.isfile(os.path.join(checkpoint_path, f)))
            if has_weights:
                result["type"] = "full_hf"
                result["hf_path"] = checkpoint_path
        return result
    
    # 检查LoRA adapter
    if os.path.exists(lora_path) and os.path.exists(os.path.join(lora_path, "adapter_config.json")):
        result["type"] = "lora"
        result["lora_path"] = lora_path
        
        # 尝试读取base_model路径
        try:
            with open(os.path.join(lora_path, "adapter_config.json"), 'r') as f:
                config = json.load(f)
                result["base_model_hint"] = config.get("base_model_name_or_path")
        except:
            pass
    
    # 检查huggingface目录是否有完整模型
    if os.path.exists(hf_path):
        # 检查是否有模型权重（不只是tokenizer）
        has_weights = any(f.endswith(('.safetensors', '.bin')) for f in os.listdir(hf_path))
        if has_weights:
            result["hf_path"] = hf_path
            # 只有真正有权重时才设置为 full_hf
            if result["type"] == "base":  # 如果之前没检测到lora
                result["type"] = "full_hf"
    
    return result


def detect_num_options_from_file(file_path):
    """Infer the number of MCQ options (2 or 4) from the prompt content."""
    try:
        df = pd.read_parquet(file_path)
        if len(df) > 0:
            first_prompt = df.iloc[0]['prompt']
            if isinstance(first_prompt, list):
                for msg in first_prompt:
                    content = msg.get('content', '')
                    # 检查是否提到C或D选项
                    if 'option C' in content or 'option D' in content or "'C'" in content or "'D'" in content:
                        return 4
                    # 检查是否明确说只有A和B
                    if 'A or B' in content and 'C' not in content and 'D' not in content:
                        return 2
        return 4  # 默认4选项
    except Exception:
        return 4


def extract_answer_from_response(response, mode, num_options=4):
    """从模型回复中提取答案"""
    valid_answers = ['A', 'B'] if num_options == 2 else ['A', 'B', 'C', 'D']
    
    if mode in ["think", "thinking"]:
        match = re.search(r'<answer>\s*([A-D])\s*</answer>', response, re.IGNORECASE)
        if match:
            answer = match.group(1).upper()
            if answer in valid_answers:
                return answer
        
        for line in reversed(response.strip().split('\n')):
            line = line.strip()
            for answer in valid_answers:
                if line == answer or line.startswith(f"{answer}.") or line.startswith(f"{answer}:"):
                    return answer
                if f"answer is {answer}" in line.upper() or f"answer: {answer}" in line.upper():
                    return answer
    
    elif mode == "direct":
        response_clean = response.strip().upper()
        if response_clean in valid_answers:
            return response_clean
        for answer in valid_answers:
            if response_clean.startswith(answer):
                return answer
        for char in response_clean:
            if char in valid_answers:
                return char
    
    return None


def extract_thinking_content(response, mode):
    """提取思考内容"""
    if mode == "think":
        pattern = r'<think>(.*?)</think>'
    elif mode == "thinking":
        pattern = r'<thinking>(.*?)</thinking>'
    else:
        return ""
    
    match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""


# ==================== 实时结果写入器 ====================
class StreamingResultWriter:
    """实时写入评估结果的类"""
    
    def __init__(self, output_path, metadata=None):
        self.output_path = output_path
        self.metadata = metadata or {}
        self.results = []
        self.correct = 0
        self.total = 0
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self._write_file()
    
    def add_result(self, result):
        result = convert_to_serializable(result)
        self.results.append(result)
        if result.get('is_correct') is True:
            self.correct += 1
        if result.get('golden_answer') is not None:
            self.total += 1
        self._write_file()
    
    def _write_file(self):
        accuracy = self.correct / self.total if self.total > 0 else 0
        output_data = {
            **self.metadata,
            "evaluated_samples": len(self.results),
            "correct": self.correct,
            "total_with_answer": self.total,
            "accuracy": accuracy,
            "results": self.results
        }
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    def get_summary(self):
        accuracy = self.correct / self.total if self.total > 0 else 0
        return {"evaluated_samples": len(self.results), "correct": self.correct, 
                "total_with_answer": self.total, "accuracy": accuracy}


# ==================== vLLM模型包装器（支持LoRA）====================
class VLLMModelWrapper:
    """使用vLLM进行批量推理的模型包装器，支持LoRA"""
    
    def __init__(self, model_path, model_type=None, tensor_parallel_size=1, 
                 lora_path=None, lora_name="default"):
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest
        from transformers import AutoTokenizer
        
        self.model_path = model_path
        self.model_type = model_type or detect_model_type(model_path)
        self.lora_path = lora_path
        self.lora_name = lora_name
        self.lora_request = None
        
        print(f"Loading model with vLLM: {model_path}")
        print(f"Model type: {self.model_type}")
        if lora_path:
            print(f"LoRA adapter: {lora_path}")
        
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # vLLM参数
        llm_kwargs = {
            "model": model_path,
            "tensor_parallel_size": tensor_parallel_size,
            "trust_remote_code": True,
            "dtype": "bfloat16",
            "max_model_len": 4096,
        }
        
        # 如果有LoRA，启用LoRA支持
        # 注意：vLLM 0.8.x 的 V1 引擎对 LoRA 支持有问题，需要使用 enforce_eager=True
        if lora_path:
            llm_kwargs["enable_lora"] = True
            # max_lora_rank is an upper bound for the vLLM engine, not the rank of
            # the actual adapter. PA-GRPO training uses lora_rank=32, so 64 is
            # more than enough; raise this only if you load a wider adapter.
            llm_kwargs["max_lora_rank"] = 64
            llm_kwargs["enforce_eager"] = True  # 禁用torch.compile，确保LoRA兼容
        
        self.llm = LLM(**llm_kwargs)
        
        # 创建LoRA请求
        if lora_path:
            self.lora_request = LoRARequest(lora_name, 1, lora_path)
        
        print("vLLM model loaded")
        
        # 预先获取答案选项的token IDs
        self._init_answer_token_ids()
    
    def _init_answer_token_ids(self):
        """初始化答案选项的token IDs（包括带空格和不带空格的版本）"""
        self.answer_token_ids = {}
        self.answer_token_ids_with_space = {}
        for answer in ['A', 'B', 'C', 'D']:
            # 不带空格的token编码
            token_ids = self.tokenizer.encode(answer, add_special_tokens=False)
            if token_ids:
                self.answer_token_ids[answer] = token_ids[-1]
            # 带空格的token编码 (如 " A" -> "ĠA")
            token_ids_space = self.tokenizer.encode(" " + answer, add_special_tokens=False)
            if token_ids_space:
                self.answer_token_ids_with_space[answer] = token_ids_space[-1]
        print(f"Answer token IDs: {self.answer_token_ids}")
        print(f"Answer token IDs (with leading space): {self.answer_token_ids_with_space}")
    
    def _extract_answer_probabilities(self, logprobs_list, num_options, response_text="", return_debug=False):
        """从logprobs中提取所有答案选项的概率
        
        对于think模式，答案在</think><answer>之后，需要找到正确的位置。
        关键改进：需要找到<answer>标签后面的token位置，而不是第一个出现A/B的位置。
        
        注意：tokenizer可能会把 ">B" 或 ">A" 合并成一个token，需要处理这种情况。
        
        Args:
            return_debug: 如果为True，返回 (answer_probs, debug_info) 元组
        """
        valid_answers = ['A', 'B'] if num_options == 2 else ['A', 'B', 'C', 'D']
        
        # 初始化概率字典和调试信息
        answer_probs = {ans: 0.0 for ans in valid_answers}
        debug_info = {
            "answer_token_idx": -1,
            "top5_tokens": [],  # 启用以便调试
            "answer_tag_found": False,
            "generated_token_at_position": None
        }
        
        if not logprobs_list or len(logprobs_list) == 0:
            if return_debug:
                return answer_probs, debug_info
            return answer_probs
        
        # 从logprobs重建生成的token序列，同时记录每个token的位置
        tokens_with_pos = []  # [(token_text, idx), ...]
        for idx, token_logprobs in enumerate(logprobs_list):
            if token_logprobs:
                # 获取实际生成的token（logprobs中logprob最高/最接近0的那个）
                # 在vLLM中，logprobs字典的顺序可能不是按logprob排序的
                # 需要找logprob最大的那个
                best_token = None
                best_logprob = float('-inf')
                for token_id, logprob_obj in token_logprobs.items():
                    if logprob_obj.logprob > best_logprob:
                        best_logprob = logprob_obj.logprob
                        best_token = logprob_obj.decoded_token
                if best_token is not None:
                    tokens_with_pos.append((best_token, idx))
        
        # 重建文本，找到<answer>标签的位置
        reconstructed_text = "".join([t[0] for t in tokens_with_pos])
        
        # 查找<answer>标签在重建文本中的位置
        answer_tag_patterns = ["<answer>", "<answer >"]
        answer_tag_pos = -1
        for pattern in answer_tag_patterns:
            pos = reconstructed_text.lower().find(pattern.lower())
            if pos >= 0:
                answer_tag_pos = pos + len(pattern)
                debug_info["answer_tag_found"] = True
                break
        
        # 找到<answer>标签后的第一个token的位置（在logprobs_list中的索引）
        answer_token_idx = -1
        if answer_tag_pos >= 0:
            # 计算累积字符数，找到超过answer_tag_pos的第一个token
            # 注意：累积长度超过answer_tag_pos时，当前token才是答案token
            cumulative_len = 0
            for token_text, idx in tokens_with_pos:
                cumulative_len += len(token_text)
                if cumulative_len > answer_tag_pos:
                    answer_token_idx = idx
                    debug_info["generated_token_at_position"] = token_text
                    break
        
        # 如果没找到<answer>标签，回退到查找最后一个A/B token的位置
        if answer_token_idx < 0:
            # 从后往前找，找到最后一个A/B token（更可能是最终答案）
            for idx in range(len(tokens_with_pos) - 1, -1, -1):
                token_text, pos_idx = tokens_with_pos[idx]
                if token_text.strip() in valid_answers:
                    answer_token_idx = pos_idx
                    debug_info["generated_token_at_position"] = token_text
                    break
        
        # 如果仍然没找到，使用原来的逻辑（第一个A/B）
        if answer_token_idx < 0:
            for idx, token_logprobs in enumerate(logprobs_list):
                if token_logprobs:
                    for token_id, logprob_obj in token_logprobs.items():
                        decoded = logprob_obj.decoded_token.strip()
                        if decoded in valid_answers:
                            answer_token_idx = idx
                            debug_info["generated_token_at_position"] = decoded
                            break
                    if answer_token_idx >= 0:
                        break
        
        debug_info["answer_token_idx"] = answer_token_idx
        
        if answer_token_idx < 0:
            if return_debug:
                return answer_probs, debug_info
            return answer_probs
        
        # 获取答案token位置的logprobs
        target_logprobs = logprobs_list[answer_token_idx]
        
        if not target_logprobs:
            if return_debug:
                return answer_probs, debug_info
            return answer_probs
        
        # 收集该位置的top5 tokens（按概率排序）
        all_tokens_at_position = []
        for token_id, logprob_obj in target_logprobs.items():
            prob = float(np.exp(logprob_obj.logprob))
            all_tokens_at_position.append({
                "token": logprob_obj.decoded_token,
                "token_id": int(token_id),
                "probability": prob,
                "logprob": float(logprob_obj.logprob)
            })
        # 按概率降序排序，取前5个
        all_tokens_at_position.sort(key=lambda x: x["probability"], reverse=True)
        debug_info["top5_tokens"] = all_tokens_at_position[:5]
        
        # 辅助函数：从token文本中提取答案字符
        def extract_answer_from_token(decoded_token):
            """从可能包含答案的token中提取答案字符
            例如: ">B" -> "B", " A" -> "A", "A" -> "A"
            """
            decoded = decoded_token.strip()
            # 直接匹配
            if decoded in valid_answers:
                return decoded
            # 检查是否以答案字符结尾（处理 ">B", ">A" 等情况）
            if len(decoded) >= 1 and decoded[-1] in valid_answers:
                return decoded[-1]
            # 检查是否包含答案字符（最后一个匹配的）
            for char in reversed(decoded):
                if char in valid_answers:
                    return char
            return None
        
        # 首先尝试通过decoded_token匹配（更可靠）
        for token_id, logprob_obj in target_logprobs.items():
            decoded = logprob_obj.decoded_token
            answer_char = extract_answer_from_token(decoded)
            if answer_char:
                prob = float(np.exp(logprob_obj.logprob))
                # 取最大概率（可能有多个匹配，如 "A" 和 " A" 或 ">A"）
                if prob > answer_probs.get(answer_char, 0.0):
                    answer_probs[answer_char] = prob
        
        # 如果通过decoded匹配找到了一些，再尝试通过token_id补充
        for answer in valid_answers:
            if answer_probs[answer] == 0.0:
                # 先尝试不带空格的token ID
                token_id = self.answer_token_ids.get(answer)
                if token_id is not None and token_id in target_logprobs:
                    answer_probs[answer] = float(np.exp(target_logprobs[token_id].logprob))
                # 再尝试带空格的token ID (如 "ĠA")
                if answer_probs[answer] == 0.0 and hasattr(self, 'answer_token_ids_with_space'):
                    token_id_space = self.answer_token_ids_with_space.get(answer)
                    if token_id_space is not None and token_id_space in target_logprobs:
                        answer_probs[answer] = float(np.exp(target_logprobs[token_id_space].logprob))
        
        if return_debug:
            return answer_probs, debug_info
        return answer_probs
    
    def generate_batch(self, prompts_list, mode="direct", max_new_tokens=2048,
                       temperature=0.0, top_p=1.0, num_options=4):
        from vllm import SamplingParams
        
        # 转换为文本prompts
        text_prompts = []
        for messages in prompts_list:
            if self.model_type == "qwen3":
                text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
                )
            else:
                text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            text_prompts.append(text)
        
        # 请求更多的logprobs以覆盖所有答案选项（设置为20以增加覆盖率）
        num_logprobs = 20
        sampling_params = SamplingParams(
            temperature=temperature, top_p=top_p, max_tokens=max_new_tokens, logprobs=num_logprobs
        )
        
        # 批量生成（带LoRA）
        if self.lora_request:
            outputs = self.llm.generate(text_prompts, sampling_params, lora_request=self.lora_request)
        else:
            outputs = self.llm.generate(text_prompts, sampling_params)
        
        results = []
        for output in outputs:
            response = output.outputs[0].text
            thinking_content = extract_thinking_content(response, mode)
            
            # 提取所有答案选项的概率（传递response以便定位答案位置），同时获取调试信息
            answer_probability, debug_info = self._extract_answer_probabilities(
                output.outputs[0].logprobs, num_options, response, return_debug=True
            )
            
            results.append({
                "response": response,
                "thinking_content": thinking_content,
                "answer_probability": answer_probability,
                "answer_debug_info": debug_info
            })
        
        return results


# ==================== Transformers模型包装器 ====================
class TransformersModelWrapper:
    """使用Transformers进行推理的模型包装器（兼容性更好）"""
    
    def __init__(self, model_path, model_type=None, base_model_path=None):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        self.model_path = model_path
        self.model_type = model_type or detect_model_type(model_path)
        self.base_model_path = base_model_path
        
        print(f"Loading model with Transformers: {model_path}")
        
        ckpt_info = detect_checkpoint_type(model_path)
        
        if ckpt_info["type"] == "lora":
            if not base_model_path and ckpt_info["base_model_hint"]:
                print(f"Auto-detected base_model: {ckpt_info['base_model_hint']}")
            self._load_lora_model(ckpt_info["lora_path"], base_model_path or ckpt_info["base_model_hint"])
        elif ckpt_info["type"] == "full_hf":
            self._load_base_model(ckpt_info["hf_path"])
        else:
            self._load_base_model(model_path)
        
        print("Model loaded")
    
    def _load_base_model(self, model_path):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side='left')
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        )
        self.model.eval()
        self._init_answer_token_ids()
    
    def _load_lora_model(self, lora_path, base_model_path):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        
        if not base_model_path:
            raise ValueError("base_model_path is required to load a LoRA model")
        
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True, padding_side='left')
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        )
        self.model = PeftModel.from_pretrained(base_model, lora_path)
        self.model = self.model.merge_and_unload()
        self.model.eval()
        self._init_answer_token_ids()
    
    def _init_answer_token_ids(self):
        """初始化答案选项的token IDs（包括带空格和不带空格的版本）"""
        self.answer_token_ids = {}
        self.answer_token_ids_with_space = {}
        for answer in ['A', 'B', 'C', 'D']:
            # 不带空格的token编码
            token_ids = self.tokenizer.encode(answer, add_special_tokens=False)
            if token_ids:
                self.answer_token_ids[answer] = token_ids[-1]
            # 带空格的token编码 (如 " A" -> "ĠA")
            token_ids_space = self.tokenizer.encode(" " + answer, add_special_tokens=False)
            if token_ids_space:
                self.answer_token_ids_with_space[answer] = token_ids_space[-1]
        print(f"Answer token IDs: {self.answer_token_ids}")
        print(f"Answer token IDs (with leading space): {self.answer_token_ids_with_space}")
    
    def _extract_answer_probabilities(self, scores, num_options, generated_tokens=None):
        """从scores中提取所有答案选项的概率
        
        对于think模式，需要找到<answer>标签后面的答案token位置，而不是第一个A/B。
        """
        valid_answers = ['A', 'B'] if num_options == 2 else ['A', 'B', 'C', 'D']
        
        # 初始化概率字典
        answer_probs = {ans: 0.0 for ans in valid_answers}
        
        if not scores or len(scores) == 0:
            return answer_probs
        
        # 先将generated_tokens解码成文本，找到<answer>标签的位置
        answer_idx = 0
        if generated_tokens is not None:
            # 解码生成的tokens
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            # 查找<answer>标签
            answer_tag_patterns = ["<answer>", "<answer >"]
            answer_tag_found = False
            for pattern in answer_tag_patterns:
                pos = generated_text.lower().find(pattern.lower())
                if pos >= 0:
                    answer_tag_found = True
                    # 找到<answer>标签后，需要确定这个位置对应哪个token
                    # 逐个token累积文本，找到超过<answer>标签位置的token
                    answer_tag_end = pos + len(pattern)
                    cumulative_text = ""
                    for idx, token_id in enumerate(generated_tokens):
                        token_text = self.tokenizer.decode([token_id], skip_special_tokens=True)
                        cumulative_text += token_text
                        if len(cumulative_text) >= answer_tag_end:
                            # 下一个token应该是答案
                            answer_idx = idx + 1 if idx + 1 < len(generated_tokens) else idx
                            break
                    break
            
            # 如果没找到<answer>标签，回退到从后往前找最后一个A/B token
            if not answer_tag_found:
                for idx in range(len(generated_tokens) - 1, -1, -1):
                    token_id = generated_tokens[idx]
                    token_id_int = token_id.item() if hasattr(token_id, 'item') else int(token_id)
                    if token_id_int in self.answer_token_ids.values() or token_id_int in self.answer_token_ids_with_space.values():
                        answer_idx = idx
                        break
        
        # 确保索引有效
        if answer_idx >= len(scores):
            answer_idx = len(scores) - 1
        if answer_idx < 0:
            answer_idx = 0
        
        # 获取答案位置的分数
        target_score = scores[answer_idx]  # shape: (batch_size, vocab_size)
        probs = torch.softmax(target_score[0], dim=-1)  # (vocab_size,)
        
        # 提取每个答案选项的概率
        for answer in valid_answers:
            prob = 0.0
            # 先尝试不带空格的token ID
            token_id = self.answer_token_ids.get(answer)
            if token_id is not None and token_id < len(probs):
                prob = float(probs[token_id].item())
            # 再尝试带空格的token ID
            if hasattr(self, 'answer_token_ids_with_space'):
                token_id_space = self.answer_token_ids_with_space.get(answer)
                if token_id_space is not None and token_id_space < len(probs):
                    prob = max(prob, float(probs[token_id_space].item()))
            answer_probs[answer] = prob
        
        return answer_probs
    
    def generate(self, messages, mode="direct", max_new_tokens=2048, temperature=0.0, top_p=1.0, num_options=4):
        if self.model_type == "qwen3":
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        else:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

        inputs = self.tokenizer(text, return_tensors="pt", padding=True).to(self.model.device)

        # Greedy when temperature == 0 (deterministic, matches the numbers in
        # the paper); otherwise sample.
        do_sample = temperature > 0.0
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
        
        new_tokens = outputs.sequences[0][inputs['input_ids'].shape[1]:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        thinking_content = extract_thinking_content(response, mode)
        
        # 提取所有答案选项的概率（传递生成的tokens以便定位答案位置）
        answer_probability = self._extract_answer_probabilities(
            outputs.scores if hasattr(outputs, 'scores') else None, 
            num_options,
            new_tokens
        )
        
        return {"response": response, "thinking_content": thinking_content, "answer_probability": answer_probability}


# ==================== 评估函数 ====================
def evaluate_dataset_batch(model, dataset_path, mode, num_options, output_path,
                           max_samples=None, batch_size=16,
                           temperature=0.0, top_p=1.0, max_new_tokens=2048):
    """使用vLLM批量评估数据集"""
    print(f"\nEvaluating dataset: {dataset_path}")

    df = pd.read_parquet(dataset_path)
    if max_samples:
        df = df.head(max_samples)

    print(f"samples: {len(df)}, batch size: {batch_size}")
    
    metadata = {"dataset": os.path.basename(dataset_path), "mode": mode, "num_options": num_options, "total_samples": len(df)}
    writer = StreamingResultWriter(output_path, metadata)
    
    # 准备所有数据
    all_data = []
    for idx, row in df.iterrows():
        messages = row['prompt']
        reward_model = row.get('reward_model', {})
        if isinstance(reward_model, str):
            reward_model = json.loads(reward_model)
        golden_answer = reward_model.get('ground_truth', None)
        
        all_data.append({
            "idx": idx, "messages": messages, "golden_answer": golden_answer,
            "data_source": row.get('data_source', ''), "ability": row.get('ability', ''),
            "extra_info": row.get('extra_info', {}),
            "user_question": next((m['content'] for m in messages if m.get('role') == 'user'), ''),
            "system_prompt": next((m['content'] for m in messages if m.get('role') == 'system'), ''),
        })
    
    for batch_start in tqdm(range(0, len(all_data), batch_size), desc="批量推理"):
        batch = all_data[batch_start:batch_start + batch_size]
        prompts = [item["messages"] for item in batch]
        outputs = model.generate_batch(
            prompts, mode=mode, num_options=num_options,
            temperature=temperature, top_p=top_p, max_new_tokens=max_new_tokens,
        )
        
        for item, output in zip(batch, outputs):
            extracted_answer = extract_answer_from_response(output['response'], mode, num_options)
            is_correct = extracted_answer == item['golden_answer'] if item['golden_answer'] else None
            
            writer.add_result({
                "row_index": item['idx'], "data_source": item['data_source'],
                "ability": item['ability'], "extra_info": item['extra_info'],
                "original_prompt": item['messages'], "user_question": item['user_question'],
                "system_prompt": item['system_prompt'], "golden_answer": item['golden_answer'],
                "model_response": output['response'], "extracted_answer": extracted_answer,
                "thinking_content": output.get('thinking_content', ''),
                "is_correct": is_correct, "answer_probability": output['answer_probability'],
                "answer_debug_info": output.get('answer_debug_info', {}),
            })
    
    summary = writer.get_summary()
    print(f"Accuracy: {summary['correct']}/{summary['total_with_answer']} = {summary['accuracy']:.4f}")
    return summary


def evaluate_dataset_sequential(model, dataset_path, mode, num_options, output_path,
                                max_samples=None,
                                temperature=0.0, top_p=1.0, max_new_tokens=2048):
    """使用Transformers逐个评估数据集"""
    print(f"\nEvaluating dataset: {dataset_path}")
    
    df = pd.read_parquet(dataset_path)
    if max_samples:
        df = df.head(max_samples)
    
    print(f"samples: {len(df)}")
    
    metadata = {"dataset": os.path.basename(dataset_path), "mode": mode, "num_options": num_options, "total_samples": len(df)}
    writer = StreamingResultWriter(output_path, metadata)
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="逐个推理"):
        try:
            messages = row['prompt']
            reward_model = row.get('reward_model', {})
            if isinstance(reward_model, str):
                reward_model = json.loads(reward_model)
            golden_answer = reward_model.get('ground_truth', None)
            
            output = model.generate(
                messages, mode=mode, num_options=num_options,
                temperature=temperature, top_p=top_p, max_new_tokens=max_new_tokens,
            )
            extracted_answer = extract_answer_from_response(output['response'], mode, num_options)
            is_correct = extracted_answer == golden_answer if golden_answer else None
            
            writer.add_result({
                "row_index": idx, "data_source": row.get('data_source', ''),
                "ability": row.get('ability', ''), "extra_info": row.get('extra_info', {}),
                "original_prompt": messages,
                "user_question": next((m['content'] for m in messages if m.get('role') == 'user'), ''),
                "system_prompt": next((m['content'] for m in messages if m.get('role') == 'system'), ''),
                "golden_answer": golden_answer, "model_response": output['response'],
                "extracted_answer": extracted_answer, "thinking_content": output.get('thinking_content', ''),
                "is_correct": is_correct, "answer_probability": output['answer_probability'],
            })
        except Exception as e:
            print(f"Error while processing sample {idx}: {e}")
            writer.add_result({"row_index": idx, "error": str(e), "golden_answer": None, "is_correct": None})
    
    summary = writer.get_summary()
    print(f"Accuracy: {summary['correct']}/{summary['total_with_answer']} = {summary['accuracy']:.4f}")
    return summary


# ==================== 主函数 ====================
def main():
    parser = argparse.ArgumentParser(description="模型测评脚本 (加速版v2 - 支持vLLM+LoRA)")
    
    # 模型参数 - 修改参数名称使其更清晰
    parser.add_argument("--model_path", type=str, default=None, 
                        help="Path to the model to evaluate (a base HF model, LoRA checkpoint, or merged HF model).")
    parser.add_argument("--base_model_path", type=str, default=None, 
                        help="Base model path; required when --model_path points at a LoRA checkpoint.")
    parser.add_argument("--model_type", type=str, choices=["llama", "qwen3", "auto"], default="auto")
    
    # 批量checkpoint评估
    parser.add_argument("--checkpoint_dir", type=str, default=None, 
                        help="Parent directory containing multiple global_step_* checkpoints to evaluate in batch.")
    parser.add_argument("--steps", type=int, nargs="+", default=None,
                        help="Specific global_step values to evaluate (default: all under --checkpoint_dir).")
    
    # 推理后端
    parser.add_argument("--use_transformers", action="store_true", help="Use Transformers instead of vLLM (slower fallback).")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    
    # 评估参数
    parser.add_argument("--mode", type=str, choices=["think", "thinking", "direct"], required=True)
    parser.add_argument("--dataset_files", type=str, nargs="+", default=None,
                        help="Specific parquet files to evaluate. If omitted, auto-discover "
                             "$PAGRPO_DATASET_DIR/*_<mode>.parquet.")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="./eval_results")

    # Decoding parameters. Default is greedy (temperature=0) to match the
    # deterministic numbers in the paper; pass --temperature > 0 to sample.
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature; 0 = greedy (default).")
    parser.add_argument("--top_p", type=float, default=1.0,
                        help="Top-p nucleus sampling threshold (ignored when temperature=0).")
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    
    args = parser.parse_args()
    
    # 确定要评估的checkpoint列表
    checkpoints_to_eval = []
    
    if args.checkpoint_dir:
        # 批量评估多个checkpoint
        for item in os.listdir(args.checkpoint_dir):
            if item.startswith("global_step_"):
                step = int(item.split("_")[-1])
                if args.steps is None or step in args.steps:
                    checkpoints_to_eval.append({
                        "path": os.path.join(args.checkpoint_dir, item),
                        "name": item,
                        "step": step
                    })
        checkpoints_to_eval.sort(key=lambda x: x["step"])
    elif args.model_path:
        checkpoints_to_eval.append({
            "path": args.model_path,
            "name": os.path.basename(args.model_path.rstrip('/')),
            "step": None
        })
    else:
        parser.error("Either --model_path or --checkpoint_dir must be specified")
    
    print("=" * 70)
    print("Evaluation config:")
    print(f"  number of models to evaluate: {len(checkpoints_to_eval)}")
    print(f"  mode: {args.mode}")
    print(f"  backend: {'transformers' if args.use_transformers else 'vLLM'}")
    print("=" * 70)
    
    # 确定要评估的数据集
    datasets_to_eval = []
    if args.dataset_files:
        for file_path in args.dataset_files:
            if os.path.exists(file_path):
                datasets_to_eval.append({
                    "name": os.path.basename(file_path).replace('.parquet', ''),
                    "path": file_path,
                    "options": detect_num_options_from_file(file_path)
                })
    else:
        # Auto-discover all `*_<mode>.parquet` files under DATASET_DIR
        pattern = os.path.join(DATASET_DIR, f"*_{args.mode}.parquet")
        for dataset_path in sorted(glob.glob(pattern)):
            datasets_to_eval.append({
                "name": os.path.basename(dataset_path).replace('.parquet', ''),
                "path": dataset_path,
                "options": detect_num_options_from_file(dataset_path),
            })
    
    if not datasets_to_eval:
        print("Error: no datasets found to evaluate")
        return

    print(f"\nWill evaluate {len(datasets_to_eval)} dataset(s)")
    
    # 对于vLLM + LoRA，可以共享base model
    shared_model = None
    
    for ckpt_info in checkpoints_to_eval:
        ckpt_path = ckpt_info["path"]
        ckpt_name = ckpt_info["name"]
        
        print(f"\n{'='*70}")
        print(f"Evaluating: {ckpt_name}")
        print(f"{'='*70}")
        
        # 创建输出目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backend = "transformers" if args.use_transformers else "vllm"
        output_subdir = os.path.join(args.output_dir, f"{ckpt_name}_{args.mode}_{backend}_{timestamp}")
        os.makedirs(output_subdir, exist_ok=True)
        
        # 检测checkpoint类型
        ckpt_type_info = detect_checkpoint_type(ckpt_path)
        model_type = args.model_type if args.model_type != "auto" else detect_model_type(ckpt_path)
        
        # 检查是否需要转换
        if ckpt_type_info.get("needs_conversion", False):
            print(f"Error: checkpoint {ckpt_name} needs conversion before it can be used; skipping.")
            continue
        
        # 加载模型
        if args.use_transformers:
            model = TransformersModelWrapper(ckpt_path, model_type=model_type, base_model_path=args.base_model_path)
            use_batch = False
        else:
            if ckpt_type_info["type"] == "lora":
                # vLLM + LoRA
                # 优先级: 命令行指定的 base_model_path > checkpoint中hint的路径 > hf_path
                # 注意: hf_path 可能只包含 tokenizer 而没有模型权重（如 verl checkpoint 的 huggingface 目录）
                if args.base_model_path:
                    base_path = args.base_model_path
                    print(f"Using base model from --base_model_path: {base_path}")
                elif ckpt_type_info.get("base_model_hint"):
                    base_path = ckpt_type_info["base_model_hint"]
                    print(f"Using base model from adapter_config.json: {base_path}")
                elif ckpt_type_info.get("hf_path"):
                    base_path = ckpt_type_info["hf_path"]
                    print(f"Using HuggingFace dir from checkpoint: {base_path}")
                else:
                    base_path = None
                
                if not base_path:
                    print("Warning: LoRA model needs --base_model_path; falling back to Transformers")
                    model = TransformersModelWrapper(ckpt_path, model_type=model_type, base_model_path=args.base_model_path)
                    use_batch = False
                else:
                    try:
                        model = VLLMModelWrapper(
                            base_path, model_type=model_type, 
                            tensor_parallel_size=args.tensor_parallel_size,
                            lora_path=ckpt_type_info["lora_path"],
                            lora_name=ckpt_name
                        )
                        use_batch = True
                    except Exception as e:
                        print(f"vLLM+LoRA load failed: {e}; falling back to Transformers")
                        model = TransformersModelWrapper(ckpt_path, model_type=model_type, base_model_path=args.base_model_path)
                        use_batch = False
            elif ckpt_type_info["type"] == "full_hf":
                # 完整HF模型
                model = VLLMModelWrapper(
                    ckpt_type_info["hf_path"], model_type=model_type,
                    tensor_parallel_size=args.tensor_parallel_size
                )
                use_batch = True
            else:
                # 普通模型
                model = VLLMModelWrapper(
                    ckpt_path, model_type=model_type,
                    tensor_parallel_size=args.tensor_parallel_size
                )
                use_batch = True
        
        # 评估每个数据集
        all_summaries = {}
        for ds_info in datasets_to_eval:
            result_file = os.path.join(output_subdir, f"{ds_info['name']}_{args.mode}_results.json")
            
            if use_batch:
                summary = evaluate_dataset_batch(
                    model, ds_info["path"], args.mode, ds_info["options"], result_file,
                    max_samples=args.max_samples, batch_size=args.batch_size,
                    temperature=args.temperature, top_p=args.top_p, max_new_tokens=args.max_new_tokens,
                )
            else:
                summary = evaluate_dataset_sequential(
                    model, ds_info["path"], args.mode, ds_info["options"], result_file,
                    max_samples=args.max_samples,
                    temperature=args.temperature, top_p=args.top_p, max_new_tokens=args.max_new_tokens,
                )
            
            all_summaries[ds_info["name"]] = summary
        
        # 保存总结
        summary_file = os.path.join(output_subdir, "summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(convert_to_serializable(all_summaries), f, ensure_ascii=False, indent=2)
        
        # 清理模型内存
        del model
        torch.cuda.empty_cache()
        
        print(f"\nResults saved to: {output_subdir}")

    print("\n" + "=" * 70)
    print("All evaluations finished.")
    print("=" * 70)


if __name__ == "__main__":
    main()
