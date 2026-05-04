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
"""Metrics for PA-GRPO's pairwise (P=2) evaluation.

Reads the per-sample JSON files produced by ``evaluate_models.py`` and reports
the paper's pairwise metrics (Section 4.2 / Table 1):
  1. Accuracy
  2. Consistency (Con)   — pair of orderings yields the flipped letter
  3. Consistent Accuracy (CA) — pair is both consistent and correct
  4. RStd  (recall standard deviation, appendix)
  5. CKLD  (class KL divergence, appendix)

Use this script for the 2-option benchmarks (MT-Bench, Chatbot Arena,
RewardBench, LLMBar, Amazon Reviews, ...). For the 4-option MCQ benchmarks
(MMLU, ARC, GPQA, ...) with 24-permutation expansion use
``compute_metrics_mcq.py`` instead.
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
    将结果按照问题ID分组，每两个为一组（原顺序和逆顺序）
    假设数据格式包含: question_id, permutation, predicted_answer, golden_answer
    """
    groups = defaultdict(list)
    
    # 先使用标准的question_id逻辑尝试分组
    for item in results:
        question_id = get_question_id(item)
        base_id = question_id.split('_perm_')[0] if '_perm_' in question_id else question_id
        groups[base_id].append(item)
    
    # 检查是否成功分组（至少有一些组有多个样本）
    groups_with_pairs = sum(1 for items in groups.values() if len(items) >= 2)
    
    # 如果几乎没有配对（<10%的组有配对），尝试使用original_index
    if groups_with_pairs < len(groups) * 0.1 and len(groups) > 1:
        # 检查是否所有样本都有original_index
        has_original_index = all('extra_info' in item and 'original_index' in item.get('extra_info', {}) 
                                 for item in results)
        
        if has_original_index:
            # 使用original_index重新分组
            groups = defaultdict(list)
            for item in results:
                base_id = str(item['extra_info']['original_index'])
                groups[base_id].append(item)
    
    return groups


def calculate_consistency(results: List[Dict]) -> Tuple[float, int, int]:
    """
    计算一致率
    每两个为一组，选项顺序相反时，如果原顺序选A逆顺序选B则一致
    
    对于MTBench数据:
    - permutation 0: 原顺序，golden可能是B
    - permutation 1: 逆顺序，golden变成A
    如果两个permutation的预测也是相反的（perm0选A，perm1选B），则一致
    
    返回: (一致率, 一致对数, 总对数)
    """
    groups = group_by_pairs(results)
    
    consistent_pairs = 0
    total_pairs = 0
    
    for base_id, items in groups.items():
        if len(items) < 2:
            continue
        
        # 检查是否只有2个样本（简单配对情况）
        if len(items) == 2:
            # 按照permutation排序
            items_sorted = sorted(items, key=lambda x: x.get('extra_info', {}).get('permutation', 0))
            
            item1 = items_sorted[0]
            item2 = items_sorted[1]
            
            pred1 = get_prediction(item1)
            pred2 = get_prediction(item2)
            golden1 = get_golden_answer(item1)
            golden2 = get_golden_answer(item2)
            
            # 检查golden是否相反（应该相反）
            golden_reversed = (golden1 != golden2)
            
            # 检查预测是否也相反
            pred_reversed = (pred1 != pred2)
            
            # 如果golden相反且预测也相反，则一致
            if golden_reversed and pred_reversed:
                consistent_pairs += 1
            
            total_pairs += 1
        else:
            # 对于有多个样本的组（如mtbench），相邻的两个样本为一对
            # 按照row_index或index排序以保持原始顺序
            items_sorted = sorted(items, key=lambda x: x.get('row_index', x.get('extra_info', {}).get('index', 0)))
            
            # 每两个相邻样本为一对
            for i in range(0, len(items_sorted) - 1, 2):
                if i + 1 >= len(items_sorted):
                    break
                    
                item1 = items_sorted[i]
                item2 = items_sorted[i + 1]
                
                pred1 = get_prediction(item1)
                pred2 = get_prediction(item2)
                golden1 = get_golden_answer(item1)
                golden2 = get_golden_answer(item2)
                
                # 检查golden是否相反（应该相反）
                golden_reversed = (golden1 != golden2)
                
                # 检查预测是否也相反
                pred_reversed = (pred1 != pred2)
                
                # 如果golden相反且预测也相反，则一致
                if golden_reversed and pred_reversed:
                    consistent_pairs += 1
                
                total_pairs += 1
    
    consistency_rate = consistent_pairs / total_pairs if total_pairs > 0 else 0.0
    return consistency_rate, consistent_pairs, total_pairs


def calculate_consistent_and_correct(results: List[Dict]) -> Tuple[float, int]:
    """
    计算一致且正确率
    只有当一对样本既一致又都正确时才计数
    
    返回: (一致且正确率, 一致且正确的对数)
    """
    groups = group_by_pairs(results)
    
    consistent_correct_pairs = 0
    total_pairs = 0
    
    for base_id, items in groups.items():
        if len(items) < 2:
            continue
        
        # 检查是否只有2个样本（简单配对情况）
        if len(items) == 2:
            # 按照permutation排序
            items_sorted = sorted(items, key=lambda x: x.get('extra_info', {}).get('permutation', 0))
            
            item1 = items_sorted[0]
            item2 = items_sorted[1]
            
            pred1 = get_prediction(item1)
            pred2 = get_prediction(item2)
            golden1 = get_golden_answer(item1)
            golden2 = get_golden_answer(item2)
            
            # 检查是否都正确
            is_correct1 = item1.get('is_correct', pred1 == golden1)
            is_correct2 = item2.get('is_correct', pred2 == golden2)
            
            # 检查是否一致（golden相反且预测也相反）
            golden_reversed = (golden1 != golden2)
            pred_reversed = (pred1 != pred2)
            is_consistent = golden_reversed and pred_reversed
            
            if is_consistent and is_correct1 and is_correct2:
                consistent_correct_pairs += 1
            
            total_pairs += 1
        else:
            # 对于有多个样本的组（如mtbench），相邻的两个样本为一对
            # 按照row_index或index排序以保持原始顺序
            items_sorted = sorted(items, key=lambda x: x.get('row_index', x.get('extra_info', {}).get('index', 0)))
            
            # 每两个相邻样本为一对
            for i in range(0, len(items_sorted) - 1, 2):
                if i + 1 >= len(items_sorted):
                    break
                    
                item1 = items_sorted[i]
                item2 = items_sorted[i + 1]
                
                pred1 = get_prediction(item1)
                pred2 = get_prediction(item2)
                golden1 = get_golden_answer(item1)
                golden2 = get_golden_answer(item2)
                
                # 检查是否都正确
                is_correct1 = item1.get('is_correct', pred1 == golden1)
                is_correct2 = item2.get('is_correct', pred2 == golden2)
                
                # 检查是否一致（golden相反且预测也相反）
                golden_reversed = (golden1 != golden2)
                pred_reversed = (pred1 != pred2)
                is_consistent = golden_reversed and pred_reversed
                
                if is_consistent and is_correct1 and is_correct2:
                    consistent_correct_pairs += 1
                
                total_pairs += 1
    
    rate = consistent_correct_pairs / total_pairs if total_pairs > 0 else 0.0
    return rate, consistent_correct_pairs


def calculate_rstd(results: List[Dict]) -> float:
    """
    计算Rstd（Recall标准差）
    
    步骤：
    1. 对每个标签，计算该标签的recall
    2. Recall = TP / (TP + FN) = 正确预测为该标签的数量 / 真实标签总数
    3. 计算所有标签recall的标准差
    """
    # 收集所有标签
    all_labels = set()
    label_true_counts = Counter()  # 真实标签计数
    label_tp_counts = Counter()     # True Positive计数
    
    for item in results:
        golden = get_golden_answer(item)
        pred = get_prediction(item)
        
        all_labels.add(golden)
        all_labels.add(pred)
        label_true_counts[golden] += 1
        
        if pred == golden:
            label_tp_counts[golden] += 1
    
    # 计算每个标签的recall
    recalls = []
    for label in all_labels:
        true_count = label_true_counts.get(label, 0)
        tp_count = label_tp_counts.get(label, 0)
        
        if true_count > 0:
            recall = tp_count / true_count
            recalls.append(recall)
    
    # 计算标准差
    if len(recalls) > 0:
        return float(np.std(recalls))
    return 0.0


def calculate_ckld(results: List[Dict]) -> float:
    """
    计算CKLD（类别KL散度）
    
    步骤：
    1. 计算每个标签的真实分布 p_i (golden_answer的频率)
    2. 计算每个标签的预测分布 q_i (预测标签的频率)
    3. 计算KL散度: CKLD = Σ p_i * log(p_i / q_i)
    """
    # 收集所有标签及其计数
    golden_counts = Counter()
    pred_counts = Counter()
    all_labels = set()
    
    for item in results:
        golden = get_golden_answer(item)
        pred = get_prediction(item)
        
        golden_counts[golden] += 1
        pred_counts[pred] += 1
        all_labels.add(golden)
        all_labels.add(pred)
    
    total = len(results)
    
    # 计算分布（加入平滑以避免除零）
    epsilon = 1e-10
    p_dist = {label: (golden_counts.get(label, 0) + epsilon) / total for label in all_labels}
    q_dist = {label: (pred_counts.get(label, 0) + epsilon) / total for label in all_labels}
    
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
Evaluation metrics report (pairwise / P=2)
{'='*80}

1. Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)
   - correct: {metrics['correct_count']}
   - total:   {metrics['total_count']}

2. Consistency (Con): {metrics['consistency']:.4f} ({metrics['consistency']*100:.2f}%)
   - consistent pairs: {metrics['consistent_pairs']}
   - total pairs:      {metrics['total_pairs']}
   Note: a pair is consistent iff the prediction flips when the option order flips.

3. Consistent Accuracy (CA): {metrics['consistent_correct']:.4f} ({metrics['consistent_correct']*100:.2f}%)
   - consistent-and-correct pairs: {metrics['consistent_correct_pairs']}

4. RStd (recall std, appendix-only): {metrics['rstd']:.4f}
   Note: standard deviation of per-label recall; lower = more balanced.

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
        print(f"错误: 路径 {args.path} 不存在！")
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
    
    # 2. 一致率
    consistency, consistent_pairs, total_pairs = calculate_consistency(results)
    
    # 3. 一致且正确率
    consistent_correct, consistent_correct_pairs = calculate_consistent_and_correct(results)
    
    # 4. Rstd
    rstd = calculate_rstd(results)

    # 5. CKLD
    ckld = calculate_ckld(results)

    # 汇总指标
    metrics = {
        'accuracy': accuracy,
        'correct_count': correct_count,
        'total_count': len(results),
        'consistency': consistency,
        'consistent_pairs': consistent_pairs,
        'total_pairs': total_pairs,
        'consistent_correct': consistent_correct,
        'consistent_correct_pairs': consistent_correct_pairs,
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
            
            # 计算指标
            accuracy = calculate_accuracy(results)
            correct_count = sum(1 for item in results if item.get('is_correct', False))
            consistency, consistent_pairs, total_pairs = calculate_consistency(results)
            consistent_correct, consistent_correct_pairs = calculate_consistent_and_correct(results)
            rstd = calculate_rstd(results)
            ckld = calculate_ckld(results)

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
                'consistent_pairs': consistent_pairs,
                'total_pairs': total_pairs,
                'consistent_correct': consistent_correct,
                'consistent_correct_pairs': consistent_correct_pairs,
                'rstd': rstd,
                'ckld': ckld,
            }
            
            all_metrics.append(metrics_row)
            
            print(f"  samples: {len(results)}, accuracy: {accuracy:.4f}, consistency: {consistency:.4f}")
            
        except Exception as e:
            print(f"  错误: {str(e)}")
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
