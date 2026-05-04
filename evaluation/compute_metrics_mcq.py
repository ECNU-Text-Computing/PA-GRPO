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
"""Metrics for PA-GRPO's MCQ evaluation with Full Permutation Expansion (P=24).

Reads the per-sample JSON files produced by ``evaluate_models.py`` for the
4-option MCQ benchmarks (MMLU-Redux, ARC-Challenge, GPQA, ...) where each
question has been expanded into all 24 permutations of its options, and reports:
  1. Accuracy
  2. Consistency (Con)   — majority-mode share of the 24 permutations
  3. Consistent Accuracy (CA) — unique mode of the 24 equals the ground truth
                                 (ties count as fragmentation, per Eq. 10)
  4. RStd  (recall standard deviation, appendix)
  5. CKLD  (class KL divergence, appendix)

For pairwise (P=2) Judge benchmarks use ``compute_metrics_judge.py`` instead.
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple
import argparse
import pandas as pd
from datetime import datetime
import os


def load_results(json_path: str) -> List[Dict]:
    """加载JSON结果文件"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 如果数据是字典且包含'results'键，提取结果列表
    if isinstance(data, dict) and 'results' in data:
        return data['results']
    # 否则假设数据本身就是列表
    elif isinstance(data, list):
        return data
    else:
        raise ValueError(f"无法识别的JSON格式。数据类型: {type(data)}, 键: {data.keys() if isinstance(data, dict) else 'N/A'}")


def get_prediction(item: Dict) -> str:
    """从结果项中提取预测答案"""
    pred = item.get('extracted_answer', 
                    item.get('predicted_answer', 
                            item.get('prediction', '')))
    # 处理None值
    if pred is None:
        return ''
    return str(pred)


def get_golden_answer(item: Dict) -> str:
    """从结果项中提取正确答案"""
    golden = item.get('golden_answer', 
                      item.get('answer', ''))
    # 处理None值
    if golden is None:
        return ''
    return str(golden)


def get_question_id(item: Dict) -> str:
    """从结果项中提取问题ID"""
    # 尝试多个可能的字段
    if 'question_id' in item:
        return str(item['question_id'])
    elif 'id' in item:
        return str(item['id'])
    elif 'extra_info' in item:
        extra = item['extra_info']
        # 优先使用question_id
        if 'question_id' in extra:
            return str(extra['question_id'])
        # judge_bench使用pair_id
        elif 'pair_id' in extra:
            return str(extra['pair_id'])
        # 其他数据集可能使用id
        elif 'id' in extra:
            return str(extra['id'])
    elif 'row_index' in item:
        return str(item['row_index'])
    return ''


def calculate_accuracy(results: List[Dict]) -> float:
    """计算正确率"""
    correct = sum(1 for item in results if item.get('is_correct', False))
    total = len(results)
    return correct / total if total > 0 else 0.0


def group_by_pairs(results: List[Dict]) -> Dict[str, List[Dict]]:
    """
    将结果按照原始问题ID分组，每个问题应该有24个不同顺序的版本
    使用 extra_info 中的 original_index 字段进行分组
    """
    groups = defaultdict(list)
    
    for item in results:
        # 优先使用 extra_info 中的 original_index
        if 'extra_info' in item and 'original_index' in item.get('extra_info', {}):
            base_id = str(item['extra_info']['original_index'])
        else:
            # 回退到使用 question_id 的方式
            question_id = get_question_id(item)
            base_id = question_id.split('_perm_')[0] if '_perm_' in question_id else question_id
        
        groups[base_id].append(item)
    
    return groups


def get_original_answer_content(item: Dict) -> str:
    """
    获取预测答案对应的原始内容（通过 permutation 映射回原始选项）
    例如：如果 permutation 是 "BACD"，预测是 "A"，则原始内容是 "B"（第一个位置）
    
    这样可以确保不同顺序下的答案可以正确比较
    """
    pred = get_prediction(item)
    if not pred or pred not in 'ABCD':
        return pred
    
    # 获取 permutation 信息
    perm = item.get('extra_info', {}).get('permutation', 'ABCD')
    if not perm or len(perm) != 4:
        return pred
    
    # 将预测的字母（A/B/C/D）映射到原始选项位置的字母
    # 例如：permutation="BACD"，预测="A" -> 原始内容是位置0的选项，即"B"
    pred_idx = ord(pred) - ord('A')  # A->0, B->1, C->2, D->3
    if 0 <= pred_idx < 4:
        return perm[pred_idx]
    return pred


def calculate_consistency_24perm(results: List[Dict]) -> Tuple[float, int, int]:
    """
    计算一致率（24顺序版本）
    
    对于每个问题组（24个顺序）：
    1. 统计每个答案内容出现的次数
    2. 找出最多一致的答案内容及其次数
    3. 一致率 = 最多一致次数 / 24
    
    最终返回所有问题组的一致率平均值
    
    返回: (平均一致率, 问题组数量, 总样本数)
    """
    groups = group_by_pairs(results)
    
    consistency_rates = []
    total_groups = 0
    
    for base_id, items in groups.items():
        if len(items) < 2:
            continue
        
        # 统计每个原始答案内容出现的次数
        answer_counts = Counter()
        for item in items:
            # 使用映射后的原始答案内容进行统计
            original_content = get_original_answer_content(item)
            answer_counts[original_content] += 1
        
        # 找出最多一致的答案数量
        if answer_counts:
            max_count = max(answer_counts.values())
            group_size = len(items)
            consistency_rate = max_count / group_size
            consistency_rates.append(consistency_rate)
            total_groups += 1
    
    # 计算平均一致率
    avg_consistency = np.mean(consistency_rates) if consistency_rates else 0.0
    return float(avg_consistency), total_groups, len(results)


def calculate_consistent_and_correct_24perm(results: List[Dict]) -> Tuple[float, int]:
    """
    计算一致且正确率（24顺序版本）
    
    对于每个问题组（24个顺序）：
    1. 找出最多一致的答案内容
    2. 判断该答案内容是否为正确答案
    3. 如果是，则该问题组"一致且正确"
    
    返回: (一致且正确率, 一致且正确的问题组数)
    """
    groups = group_by_pairs(results)
    
    consistent_correct_count = 0
    total_groups = 0
    
    for base_id, items in groups.items():
        if len(items) < 2:
            continue
        
        # 统计每个原始答案内容出现的次数
        answer_counts = Counter()
        for item in items:
            original_content = get_original_answer_content(item)
            answer_counts[original_content] += 1
        
        if not answer_counts:
            continue

        # Per Eq. (10) of the paper, fragmentation (no UNIQUE mode) counts as
        # inconsistent. If the top-2 answer counts are tied, there is no unique
        # mode and we skip awarding CA — the group is just counted as not
        # consistent-and-correct.
        ranked = answer_counts.most_common(2)
        if len(ranked) >= 2 and ranked[0][1] == ranked[1][1]:
            total_groups += 1
            continue

        # 找出最多一致的答案内容
        most_common_answer = ranked[0][0]

        # 检查这个最多一致的答案是否是正确答案
        # 正确答案在所有 permutation 下映射后应该都是相同的原始选项
        # 我们使用第一个样本来获取正确的原始答案
        first_item = items[0]
        original_answer_idx = first_item.get('extra_info', {}).get('original_answer_idx', None)
        
        if original_answer_idx is not None:
            # 原始正确答案的字母 (0->A, 1->B, 2->C, 3->D)
            correct_original = chr(ord('A') + original_answer_idx)
        else:
            # 回退方案：使用第一个样本的 golden_answer 映射到原始选项
            golden = get_golden_answer(first_item)
            perm = first_item.get('extra_info', {}).get('permutation', 'ABCD')
            if golden and golden in 'ABCD' and perm and len(perm) == 4:
                golden_idx = ord(golden) - ord('A')
                correct_original = perm[golden_idx] if 0 <= golden_idx < 4 else golden
            else:
                correct_original = golden
        
        # 判断最多一致的答案是否为正确答案
        if most_common_answer == correct_original:
            consistent_correct_count += 1
        
        total_groups += 1
    
    rate = consistent_correct_count / total_groups if total_groups > 0 else 0.0
    return float(rate), consistent_correct_count


def calculate_rstd_24perm(results: List[Dict]) -> float:
    """
    计算Rstd（Recall标准差）- 24顺序版本
    
    基于每个问题组24个顺序中各个原始选项（A/B/C/D）被选择的比例
    计算这些选择比例的标准差，反映模型对不同位置/选项的偏好差异
    
    步骤：
    1. 对每个问题组，统计24个顺序中每个原始选项被选择的次数
    2. 计算整体上各个原始选项被选择的频率
    3. 计算这些频率的标准差
    """
    groups = group_by_pairs(results)
    
    # 统计所有问题组中各个原始选项被选择的总次数
    total_answer_counts = Counter()  # 统计每个原始选项被选择的总次数
    total_samples = 0
    
    for base_id, items in groups.items():
        if len(items) < 2:
            continue
        
        for item in items:
            original_content = get_original_answer_content(item)
            if original_content in 'ABCD':
                total_answer_counts[original_content] += 1
                total_samples += 1
    
    if total_samples == 0:
        return 0.0
    
    # 计算每个原始选项的选择频率
    frequencies = []
    for label in ['A', 'B', 'C', 'D']:
        freq = total_answer_counts.get(label, 0) / total_samples
        frequencies.append(freq)
    
    # 计算标准差
    return float(np.std(frequencies))


def calculate_ckld_24perm(results: List[Dict]) -> float:
    """
    计算CKLD（类别KL散度）- 24顺序版本
    
    计算真实答案分布与预测答案分布之间的KL散度
    使用原始选项（映射后的答案）进行统计
    
    步骤：
    1. 计算每个原始选项的真实分布 p_i
    2. 计算每个原始选项的预测分布 q_i
    3. 计算KL散度: CKLD = Σ p_i * log(p_i / q_i)
    """
    groups = group_by_pairs(results)
    
    # 统计真实答案和预测答案的分布
    golden_counts = Counter()
    pred_counts = Counter()
    total = 0
    
    for base_id, items in groups.items():
        if len(items) < 2:
            continue
        
        # 获取该问题组的正确原始答案
        first_item = items[0]
        original_answer_idx = first_item.get('extra_info', {}).get('original_answer_idx', None)
        
        if original_answer_idx is not None:
            correct_original = chr(ord('A') + original_answer_idx)
        else:
            golden = get_golden_answer(first_item)
            perm = first_item.get('extra_info', {}).get('permutation', 'ABCD')
            if golden and golden in 'ABCD' and perm and len(perm) == 4:
                golden_idx = ord(golden) - ord('A')
                correct_original = perm[golden_idx] if 0 <= golden_idx < 4 else golden
            else:
                correct_original = golden
        
        # 统计每个样本的预测
        for item in items:
            original_content = get_original_answer_content(item)
            if original_content in 'ABCD':
                pred_counts[original_content] += 1
                golden_counts[correct_original] += 1
                total += 1
    
    if total == 0:
        return 0.0
    
    # 计算分布（加入平滑以避免除零）
    epsilon = 1e-10
    all_labels = ['A', 'B', 'C', 'D']
    
    p_dist = {label: (golden_counts.get(label, 0) + epsilon) / (total + epsilon * len(all_labels)) 
              for label in all_labels}
    q_dist = {label: (pred_counts.get(label, 0) + epsilon) / (total + epsilon * len(all_labels)) 
              for label in all_labels}
    
    # 计算KL散度
    kl_div = 0.0
    for label in all_labels:
        p_i = p_dist[label]
        q_i = q_dist[label]
        kl_div += p_i * np.log(p_i / q_i)
    
    return float(kl_div)


def print_metrics_report(metrics: Dict, save_path: str = None):
    """Print (and optionally save) a formatted metrics report."""
    report = f"""
{'='*80}
Evaluation metrics report (MCQ / Full Permutation Expansion, P=24)
{'='*80}

1. Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)
   - correct: {metrics['correct_count']}
   - total:   {metrics['total_count']}

2. Consistency (Con): {metrics['consistency']:.4f} ({metrics['consistency']*100:.2f}%)
   - question groups: {metrics['total_groups']}
   Note: per group, share of the most-common canonical answer across the 24 permutations.

3. Consistent Accuracy (CA): {metrics['consistent_correct']:.4f} ({metrics['consistent_correct']*100:.2f}%)
   - consistent-and-correct groups: {metrics['consistent_correct_count']}
   Note: unique mode of the group equals the ground-truth; ties (fragmentation) count as incorrect.

4. RStd (selection std, appendix-only): {metrics['rstd']:.4f}
   Note: standard deviation of the A/B/C/D selection frequency; lower = more balanced.

5. CKLD (class KL divergence, appendix-only): {metrics['ckld']:.4f}
   Note: KL(ground_truth_dist || predicted_dist); lower = closer to the true distribution.

{'='*80}
"""

    print(report)

    if save_path:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\nReport saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='计算评估指标')
    parser.add_argument('path', type=str, help='结果JSON文件路径或包含结果文件的目录')
    parser.add_argument('--output', '-o', type=str, default=None, 
                        help='输出报告文件路径（可选）')
    parser.add_argument('--save-json', type=str, default=None,
                        help='保存指标为JSON文件（可选）')
    parser.add_argument('--save-excel', type=str, default=None,
                        help='保存指标为Excel文件（可选，处理目录时自动生成）')
    parser.add_argument('--recursive', '-r', action='store_true', default=True,
                        help='递归搜索目录下的所有results.json文件（默认开启）')
    parser.add_argument('--no-recursive', action='store_false', dest='recursive',
                        help='不递归搜索子目录')
    
    args = parser.parse_args()
    
    # 判断是文件还是目录
    path = Path(args.path)
    
    if path.is_file():
        # 单个文件处理
        process_single_file(args.path, args.output, args.save_json)
    elif path.is_dir():
        # 目录批量处理
        process_directory(args.path, args.save_excel, args.recursive)
    else:
        print(f"Error: path {args.path} does not exist")
        return


def process_single_file(json_path: str, output_path: str = None, save_json_path: str = None):
    """处理单个JSON文件"""
    # 加载结果
    print(f"正在加载结果文件: {json_path}")
    results = load_results(json_path)
    print(f"加载了 {len(results)} 条结果")
    
    # 计算所有指标
    print("\n开始计算指标...")
    
    # 1. 正确率
    accuracy = calculate_accuracy(results)
    correct_count = sum(1 for item in results if item.get('is_correct', False))
    
    # 2. 一致率（24顺序版本）
    consistency, total_groups, total_samples = calculate_consistency_24perm(results)
    
    # 3. 一致且正确率（24顺序版本）
    consistent_correct, consistent_correct_count = calculate_consistent_and_correct_24perm(results)
    
    # 4. Rstd（24顺序版本）
    rstd = calculate_rstd_24perm(results)
    
    # 5. CKLD（24顺序版本）
    ckld = calculate_ckld_24perm(results)
    
    # 汇总指标
    metrics = {
        'accuracy': accuracy,
        'correct_count': correct_count,
        'total_count': len(results),
        'consistency': consistency,
        'total_groups': total_groups,
        'consistent_correct': consistent_correct,
        'consistent_correct_count': consistent_correct_count,
        'rstd': rstd,
        'ckld': ckld,
    }
    
    # 打印报告
    print_metrics_report(metrics, output_path)
    
    # 保存JSON
    if save_json_path:
        with open(save_json_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"指标已保存为JSON: {save_json_path}")


def process_directory(dir_path: str, excel_path: str = None, recursive: bool = True):
    """批量处理目录下的所有results.json文件"""
    dir_path = Path(dir_path)
    
    # 搜索所有results.json文件
    if recursive:
        pattern = "**/*results.json"
    else:
        pattern = "*results.json"
    
    json_files = list(dir_path.glob(pattern))
    
    if not json_files:
        print(f"No results.json files found under {dir_path}")
        return
    
    print(f"Found {len(json_files)} results.json files")
    print("="*80)
    
    all_metrics = []
    
    for json_file in sorted(json_files):
        print(f"\nProcessing: {json_file.relative_to(dir_path)}")
        
        try:
            # 加载结果
            results = load_results(str(json_file))
            
            # 计算指标（24顺序版本）
            accuracy = calculate_accuracy(results)
            correct_count = sum(1 for item in results if item.get('is_correct', False))
            consistency, total_groups, total_samples = calculate_consistency_24perm(results)
            consistent_correct, consistent_correct_count = calculate_consistent_and_correct_24perm(results)
            rstd = calculate_rstd_24perm(results)
            ckld = calculate_ckld_24perm(results)
            
            # 提取文件信息
            relative_path = json_file.relative_to(dir_path)
            parts = relative_path.parts
            
            # 尝试提取有用的信息
            parent_dir = parts[0] if len(parts) > 0 else ''
            filename = json_file.stem
            
            # 尝试从文件名或目录名提取checkpoint信息
            checkpoint = ''
            dataset = ''
            mode = ''
            
            # 从父目录名提取信息
            if 'global_step' in parent_dir:
                # 如: global_step_200_merged_direct_vllm_20251219_235041
                checkpoint = parent_dir.split('_vllm')[0] if '_vllm' in parent_dir else parent_dir
            elif 'Meta-Llama' in parent_dir or 'Llama' in parent_dir or 'llama' in parent_dir:
                # 如: Meta-Llama-3___1-8B-Instruct_direct_vllm_20251219_190453
                checkpoint = parent_dir.split('_vllm')[0] if '_vllm' in parent_dir else parent_dir
            else:
                # 使用父目录名作为checkpoint
                checkpoint = parent_dir.split('_vllm')[0] if '_vllm' in parent_dir else parent_dir
            
            # 从文件名提取数据集和模式信息
            if '_results' in filename:
                base_name = filename.replace('_results', '')
                parts_name = base_name.split('_')
                
                # 提取数据集名称
                if 'mtbench' in base_name:
                    dataset = 'mtbench'
                elif 'judge_bench' in base_name or 'judge-bench' in base_name:
                    dataset = 'judge_bench'
                elif 'preference_bench' in base_name or 'preference-bench' in base_name:
                    dataset = 'preference_bench'
                elif 'rewardbench' in base_name:
                    dataset = 'rewardbench'
                elif 'mmlu' in base_name:
                    dataset = 'mmlu'
                else:
                    dataset = parts_name[0] if parts_name else 'unknown'
                
                # 提取模式（direct/think）
                if 'direct' in base_name:
                    mode = 'direct'
                elif 'think' in base_name:
                    mode = 'think'
            
            metrics_row = {
                'checkpoint': checkpoint,
                'dataset': dataset,
                'mode': mode,
                'file_path': str(relative_path),
                'total_samples': len(results),
                'correct_count': correct_count,
                'accuracy': accuracy,
                'consistency': consistency,
                'total_groups': total_groups,
                'consistent_correct': consistent_correct,
                'consistent_correct_count': consistent_correct_count,
                'rstd': rstd,
                'ckld': ckld,
            }
            
            all_metrics.append(metrics_row)
            
            print(f"  samples: {len(results)}, accuracy: {accuracy:.4f}, consistency: {consistency:.4f}")

        except Exception as e:
            print(f"  Error: {str(e)}")
            continue
    
    if not all_metrics:
        print("\nNo files processed successfully")
        return
    
    # 创建DataFrame
    df = pd.DataFrame(all_metrics)
    
    # 排序
    df = df.sort_values(['checkpoint', 'dataset', 'mode'])
    
    # 生成Excel文件路径
    if excel_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        excel_path = dir_path / f"metrics_summary_{timestamp}.xlsx"
    
    # 保存为Excel
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # 主表：所有指标
        df.to_excel(writer, sheet_name='all_metrics', index=False)

        # 格式化列宽
        worksheet = writer.sheets['all_metrics']
        for idx, col in enumerate(df.columns):
            max_length = max(
                df[col].astype(str).map(len).max(),
                len(col)
            )
            worksheet.column_dimensions[chr(65 + idx)].width = min(max_length + 2, 50)
        
        # 按checkpoint分组的汇总
        if 'checkpoint' in df.columns and df['checkpoint'].notna().any():
            summary_by_checkpoint = df.groupby('checkpoint').agg({
                'dataset': 'count',
                'accuracy': 'mean',
                'consistency': 'mean',
                'consistent_correct': 'mean',
                'rstd': 'mean',
                'ckld': 'mean',
            }).round(4)
            summary_by_checkpoint.columns = ['num_datasets', 'mean_accuracy', 'mean_consistency',
                                              'mean_consistent_correct', 'mean_rstd', 'mean_ckld']
            summary_by_checkpoint.to_excel(writer, sheet_name='by_checkpoint')
        
        # 按数据集分组的汇总
        if 'dataset' in df.columns and df['dataset'].notna().any():
            summary_by_dataset = df.groupby('dataset').agg({
                'checkpoint': 'count',
                'accuracy': 'mean',
                'consistency': 'mean',
                'consistent_correct': 'mean',
                'rstd': 'mean',
                'ckld': 'mean',
            }).round(4)
            summary_by_dataset.columns = ['num_checkpoints', 'mean_accuracy', 'mean_consistency',
                                           'mean_consistent_correct', 'mean_rstd', 'mean_ckld']
            summary_by_dataset.to_excel(writer, sheet_name='by_dataset')
    
    print("\n"+"="*80)
    print(f"Processed {len(all_metrics)} files successfully")
    print(f"Saved to: {excel_path}")
    print("="*80)


if __name__ == '__main__':
    main()