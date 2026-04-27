from __future__ import annotations

import argparse
import csv
import re
from io import StringIO
from pathlib import Path


STANDARD_COLUMNS = {"类别", "类别内排名", "图片文件名", "分数差异"}


def parse_model_pair(csv_path: Path) -> tuple[str, str]:
	match = re.match(r"(.+)_vs_(.+)_top_\d+_by_category\.csv$", csv_path.name)
	if not match:
		return "", ""
	return match.group(1), match.group(2)


def safe_float(value: str | None) -> float:
	if value is None:
		return float("-inf")
	text = value.strip()
	if not text:
		return float("-inf")
	try:
		return float(text)
	except ValueError:
		return float("-inf")


def read_csv_with_fallback(csv_path: Path) -> csv.DictReader:
	for encoding in ("utf-8-sig", "utf-8", "gbk", "cp936"):
		try:
			text = csv_path.read_text(encoding=encoding)
		except UnicodeDecodeError:
			continue
		reader = csv.DictReader(StringIO(text))
		if reader.fieldnames is not None:
			return reader
	raise UnicodeDecodeError("utf-8", b"", 0, 1, f"无法识别文件编码: {csv_path}")


def build_output_row(
	row: dict[str, str],
	source_file: str,
	model_left: str,
	model_right: str,
	model_left_column: str,
	model_right_column: str,
) -> dict[str, str]:
	return {
		"来源文件": source_file,
		"模型1": model_left,
		"模型2": model_right,
		"类别": row.get("类别", "").strip(),
		"类别内排名": row.get("类别内排名", "").strip(),
		"图片文件名": row.get("图片文件名", "").strip(),
		"分数差异": row.get("分数差异", "").strip(),
		"模型1分数": row.get(model_left_column, "").strip(),
		"模型2分数": row.get(model_right_column, "").strip(),
	}


def sort_key(row: dict[str, str]) -> tuple[str, str, int]:
	try:
		rank = int(row.get("类别内排名", "") or 10**9)
	except ValueError:
		rank = 10**9
	return (
		row.get("类别", ""),
		row.get("图片文件名", ""),
		rank,
	)


def main() -> int:
	parser = argparse.ArgumentParser(description="汇总 pairwise_results 目录下的 CSV 文件并去重。")
	parser.add_argument(
		"--input-dir",
		type=Path,
		default=Path(__file__).resolve().parent / "pairwise_results_1",
		help="包含 pairwise CSV 文件的目录。",
	)
	parser.add_argument(
		"--output",
		type=Path,
		default=Path(__file__).resolve().parent / "pairwise_results_1_merged_dedup.csv",
		help="输出总表路径。",
	)
	args = parser.parse_args()

	input_dir = args.input_dir
	if not input_dir.is_dir():
		raise FileNotFoundError(f"输入目录不存在: {input_dir}")

	rows_by_key: dict[tuple[str, str], dict[str, str]] = {}
	source_files_by_key: dict[tuple[str, str], set[str]] = {}
	processed_files = 0
	processed_rows = 0

	for csv_path in sorted(input_dir.glob("*.csv")):
		if csv_path.resolve() == args.output.resolve():
			continue

		model_left, model_right = parse_model_pair(csv_path)
		reader = read_csv_with_fallback(csv_path)
		if reader.fieldnames is None:
			continue

		fieldnames = [name.strip() for name in reader.fieldnames]
		model_columns = [name for name in fieldnames if name not in STANDARD_COLUMNS]
		if len(model_columns) < 2:
			raise ValueError(f"文件列数不足，无法识别模型分数列: {csv_path}")

		model_left_column = model_columns[0]
		model_right_column = model_columns[1]
		processed_files += 1

		for row in reader:
			processed_rows += 1
			category = row.get("类别", "").strip()
			image_name = row.get("图片文件名", "").strip()
			key = (category, image_name)
			output_row = build_output_row(
				row=row,
				source_file=csv_path.name,
				model_left=model_left,
				model_right=model_right,
				model_left_column=model_left_column,
				model_right_column=model_right_column,
			)

			previous = rows_by_key.get(key)
			if previous is None:
				rows_by_key[key] = output_row
				source_files_by_key[key] = {csv_path.name}
				continue

			source_files_by_key[key].add(csv_path.name)
			current_score = safe_float(output_row.get("分数差异"))
			previous_score = safe_float(previous.get("分数差异"))
			if current_score > previous_score:
				rows_by_key[key] = output_row

	output_rows = list(rows_by_key.values())
	output_rows.sort(key=sort_key)

	with args.output.open("w", newline="", encoding="utf-8-sig") as handle:
		writer = csv.DictWriter(
			handle,
			fieldnames=[
				"来源文件",
				"类别",
				"图片文件名",
				"类别内排名",
				"分数差异",
				"模型1",
				"模型1分数",
				"模型2",
				"模型2分数",
				"来源文件数",
				"来源文件列表",
			],
		)
		writer.writeheader()
		for row in output_rows:
			key = (row["类别"], row["图片文件名"])
			row = dict(row)
			row["来源文件数"] = str(len(source_files_by_key.get(key, set())))
			row["来源文件列表"] = ";".join(sorted(source_files_by_key.get(key, set())))
			writer.writerow(row)

	print(f"已处理 {processed_files} 个 CSV，读取 {processed_rows} 行，去重后保留 {len(output_rows)} 行。")
	print(f"输出文件: {args.output}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
