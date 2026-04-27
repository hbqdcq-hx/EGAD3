from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_CSV = BASE_DIR / "pairwise_results_1_mae_matrix.csv"
DEFAULT_OUTPUT_PDF = BASE_DIR /  "pairwise_results_1_mae_heatmap.pdf"
DEFAULT_OUTPUT_CSV = BASE_DIR /  "pairwise_results_1_mae_ratio_matrix.csv"


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Build a pairwise MAE heatmap from pairwise_results_50_mae_matrix.csv.")
	parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV, help="Input MAE matrix CSV path")
	parser.add_argument("--output-pdf", type=Path, default=DEFAULT_OUTPUT_PDF, help="Output heatmap PDF path")
	parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV, help="Output ratio matrix CSV path")
	return parser.parse_args()


def load_matrix(input_csv: Path) -> pd.DataFrame:
	if not input_csv.is_file():
		raise FileNotFoundError(f"Input CSV not found: {input_csv}")

	matrix = pd.read_csv(input_csv, index_col=0)
	matrix = matrix.apply(pd.to_numeric, errors="coerce")
	return matrix


def build_ratio_matrix(matrix: pd.DataFrame) -> pd.DataFrame:
	ratio_matrix = matrix.divide(matrix.T)
	ratio_matrix = ratio_matrix.where(matrix.notna() & matrix.T.notna())
	if ratio_matrix.shape[0] == ratio_matrix.shape[1]:
		np.fill_diagonal(ratio_matrix.values, 0)
	return ratio_matrix


def save_heatmap(matrix: pd.DataFrame, output_pdf: Path) -> None:
	output_pdf.parent.mkdir(parents=True, exist_ok=True)

	plt.rcParams["font.family"] = ["Times New Roman"]
	plt.rcParams["font.serif"] = ["Times New Roman"]
	plt.rcParams["font.sans-serif"] = ["Times New Roman"]
	plt.rcParams["axes.unicode_minus"] = False
	plt.rcParams["pdf.fonttype"] = 42
	plt.rcParams["ps.fonttype"] = 42
	plt.rcParams["axes.titlesize"] = 20
	plt.rcParams["axes.labelsize"] = 20
	plt.rcParams["xtick.labelsize"] = 20
	plt.rcParams["ytick.labelsize"] = 20
	plt.rcParams["font.size"] = 15

	fig, ax = plt.subplots(figsize=(16, 13))
	blue_cmap = sns.light_palette("#1f77b4", as_cmap=True)
	sns.heatmap(
		matrix,
		ax=ax,
		cmap=blue_cmap,
		annot=True,
		fmt=".3f",
		linewidths=0.5,
		linecolor="#d7ead1",
		mask=matrix.isna(),
		annot_kws={"size": 16, "family": "Times New Roman", "color": "black"},
	)
	#ax.set_title("Pairwise MAE Ratio Heatmap", pad=16, color="#1f4f7a")
	#ax.set_xlabel("Model b")
	#ax.set_ylabel("Model a")
	plt.xticks(rotation=45, ha="right")
	plt.yticks(rotation=0)
	ax.tick_params(axis="both", labelsize=15)
	colorbar = ax.collections[0].colorbar
	colorbar.ax.tick_params(labelsize=16, labelcolor="black")
	fig.tight_layout()
	fig.savefig(output_pdf, format="pdf", bbox_inches="tight")
	plt.close(fig)


def main() -> None:
	args = parse_args()
	mae_matrix = load_matrix(args.input_csv)
	ratio_matrix = build_ratio_matrix(mae_matrix)
	ratio_matrix.to_csv(args.output_csv, encoding="utf-8-sig", float_format="%.3f")
	save_heatmap(ratio_matrix, args.output_pdf)
	print(f"Saved ratio matrix to {args.output_csv}")
	print(f"Saved heatmap to {args.output_pdf}")
	print(ratio_matrix.round(3).to_string(float_format=lambda value: f"{value:.3f}"))


if __name__ == "__main__":
	main()
