#!/usr/bin/env python3
"""
比较R目录下模型之间的预测分数差异
按类别找出前两百个差异最大的图片名字，并用两个模型名称命名输出文件
"""

import os
import csv
import itertools
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

# 数据目录
DATA_ROOT = Path("R")

# 所有类别
CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper"
]


def sanitize_filename_part(value: str) -> str:
    return value.replace(" ", "_").replace(os.sep, "_")


def natural_sort_key(value: str):
    parts = re.split(r"(\d+)", value)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def get_model_dirs(data_root: Path) -> list[Path]:
    return sorted(path for path in data_root.iterdir() if path.is_dir())


def find_score_file(model_dir: Path, category: str) -> Path | None:
    preferred_names = [
        f"anomaly_scores_{category}.csv",
        f"anomaly_scores_mvtec_{category}.csv",
    ]

    for name in preferred_names:
        candidate = model_dir / name
        if candidate.is_file():
            return candidate

    return None


def model_has_all_scores(model_dir: Path) -> bool:
    return all(find_score_file(model_dir, category) is not None for category in CATEGORIES)

def load_model_scores(model_dir: str) -> Dict[str, Dict[str, float]]:
    """
    加载模型的所有类别分数
    返回字典：{category: {filename: score}}
    """
    model_scores = {}
    
    for category in CATEGORIES:
        csv_path = find_score_file(Path(model_dir), category)
        if csv_path is None:
            print(f"警告: 文件不存在: {model_dir}/anomaly_scores_{category}.csv")
            continue
            
        category_scores = {}
        try:
            with csv_path.open('r', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader)  # 跳过标题行
                
                # 检查列名
                if "File_Path" in header[0]:
                    filename_col = 0
                    score_col = 1
                elif "File_Name" in header[0]:
                    filename_col = 0
                    score_col = 1
                else:
                    # 默认第一列是文件名，第二列是分数
                    filename_col = 0
                    score_col = 1
                
                for row in reader:
                    if len(row) >= 2:
                        filename = row[filename_col].strip()
                        try:
                            score = float(row[score_col].strip())
                            category_scores[filename] = score
                        except ValueError:
                            print(f"警告: 无法解析分数: {row[score_col]} 在文件 {csv_path}")
        except Exception as e:
            print(f"错误: 读取文件 {csv_path} 时出错: {e}")
            continue
            
        model_scores[category] = category_scores
    
    return model_scores


def compare_model_scores(
    model1_scores: Dict[str, Dict[str, float]],
    model2_scores: Dict[str, Dict[str, float]],
) -> List[Tuple[str, str, float]]:
    """
    比较两个模型的分数差异
    返回列表: [(category, filename, absolute_difference), ...]
    """
    differences = []

    for category in CATEGORIES:
        if category not in model1_scores or category not in model2_scores:
            print(f"警告: 类别 {category} 在其中一个模型中不存在")
            continue

        scores1 = model1_scores[category]
        scores2 = model2_scores[category]

        common_filenames = sorted(
            set(scores1.keys()) & set(scores2.keys()),
            key=natural_sort_key,
        )
        print(f"类别 {category}: 共有 {len(common_filenames)} 张图片")

        for filename in common_filenames:
            diff = abs(scores1[filename] - scores2[filename])
            differences.append((category, filename, diff))

    return differences


def save_top_differences_by_category(
    differences: List[Tuple[str, str, float]],
    model1_scores: Dict[str, Dict[str, float]],
    model2_scores: Dict[str, Dict[str, float]],
    model1_name: str,
    model2_name: str,
    output_dir: Path,
    top_n: int = 1,
):
    """
    按类别分别保存前N个差异最大的图片，并在每个类别内按差异降序排序。
    """
    diffs_by_category = defaultdict(list)
    for category, filename, diff in differences:
        diffs_by_category[category].append((filename, diff))

    output_file = output_dir / f"{model1_name}_vs_{model2_name}_top_{top_n}_by_category.csv"

    with output_file.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["类别", "类别内排名", "图片文件名", "分数差异", f"{model1_name}分数", f"{model2_name}分数"])

        total_rows = 0
        for category in CATEGORIES:
            if category not in diffs_by_category:
                continue

            sorted_diffs = sorted(
                diffs_by_category[category],
                key=lambda item: (item[1], item[0]),
                reverse=True,
            )

            for rank, (filename, diff) in enumerate(sorted_diffs[:top_n], 1):
                score1 = model1_scores[category][filename]
                score2 = model2_scores[category][filename]
                writer.writerow([
                    category,
                    rank,
                    filename,
                    f"{diff:.10f}",
                    f"{score1:.10f}",
                    f"{score2:.10f}",
                ])
                total_rows += 1

    print(f"\n已保存按类别排序后的前 {top_n} 个差异到 {output_file}，共 {total_rows} 行")


def analyze_by_category(differences: List[Tuple[str, str, float]]):
    """
    按类别分析差异
    """
    category_stats = {}
    
    for category, filename, diff in differences:
        if category not in category_stats:
            category_stats[category] = {
                'count': 0,
                'total_diff': 0.0,
                'max_diff': 0.0,
                'max_diff_file': ''
            }
        
        stats = category_stats[category]
        stats['count'] += 1
        stats['total_diff'] += diff
        
        if diff > stats['max_diff']:
            stats['max_diff'] = diff
            stats['max_diff_file'] = filename
    
    print("\n按类别统计:")
    print("-" * 80)
    print(f"{'类别':<12} {'图片数量':<10} {'平均差异':<15} {'最大差异':<15} {'最大差异图片'}")
    print("-" * 80)
    
    for category in sorted(category_stats.keys()):
        stats = category_stats[category]
        avg_diff = stats['total_diff'] / stats['count'] if stats['count'] > 0 else 0
        print(f"{category:<12} {stats['count']:<10} {avg_diff:<15.10f} {stats['max_diff']:<15.10f} {stats['max_diff_file'][:30]}")


def save_all_differences(
    differences: List[Tuple[str, str, float]],
    model1_scores: Dict[str, Dict[str, float]],
    model2_scores: Dict[str, Dict[str, float]],
    model1_name: str,
    model2_name: str,
    output_dir: Path,
):
    output_file = output_dir / f"{model1_name}_vs_{model2_name}_all_differences.csv"

    sorted_differences = sorted(
        differences,
        key=lambda item: (CATEGORIES.index(item[0]), natural_sort_key(item[1])),
    )

    with output_file.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["类别", "图片文件名", "分数差异", f"{model1_name}分数", f"{model2_name}分数"])

        for category, filename, diff in sorted_differences:
            score1 = model1_scores[category][filename]
            score2 = model2_scores[category][filename]
            writer.writerow([
                category,
                filename,
                f"{diff:.10f}",
                f"{score1:.10f}",
                f"{score2:.10f}",
            ])

    print(f"已保存所有 {len(sorted_differences)} 张图片的差异到 {output_file}")


def compare_models(model1_dir: Path, model2_dir: Path, output_dir: Path):
    model1_name = sanitize_filename_part(model1_dir.name)
    model2_name = sanitize_filename_part(model2_dir.name)

    print(f"\n开始比较 {model1_name} 和 {model2_name} 模型的预测分数差异...")
    print(f"\n加载 {model1_name} 模型分数...")
    model1_scores = load_model_scores(str(model1_dir))
    print(f"\n加载 {model2_name} 模型分数...")
    model2_scores = load_model_scores(str(model2_dir))

    print("\n比较两个模型的分数差异...")
    differences = compare_model_scores(model1_scores, model2_scores)

    print(f"\n总共比较了 {len(differences)} 张图片")
    if len(differences) == 0:
        print("错误: 没有找到可比较的图片")
        return

    analyze_by_category(differences)

    save_top_differences_by_category(
        differences,
        model1_scores,
        model2_scores,
        model1_name=model1_name,
        model2_name=model2_name,
        output_dir=output_dir,
        top_n=1,
    )

    # save_all_differences(
    #     differences,
    #     model1_scores,
    #     model2_scores,
    #     model1_name,
    #     model2_name,
    #     output_dir,
    # )

if __name__ == "__main__":
    if not DATA_ROOT.is_dir():
        raise FileNotFoundError(f"数据目录不存在: {DATA_ROOT}")

    model_dirs = get_model_dirs(DATA_ROOT)
    complete_model_dirs = [model_dir for model_dir in model_dirs if model_has_all_scores(model_dir)]
    skipped_model_dirs = [model_dir.name for model_dir in model_dirs if model_dir not in complete_model_dirs]

    if skipped_model_dirs:
        print(f"跳过以下不完整模型目录: {', '.join(skipped_model_dirs)}")

    if len(complete_model_dirs) < 2:
        raise ValueError("data 目录下至少需要两个完整模型目录才能进行两两对比")

    output_dir = Path("pairwise_results")
    output_dir.mkdir(exist_ok=True)

    for model1_dir, model2_dir in itertools.combinations(complete_model_dirs, 2):
        compare_models(model1_dir, model2_dir, output_dir)

    print("\n完成！")
