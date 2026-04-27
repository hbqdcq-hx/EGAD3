from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_CSV = BASE_DIR / "pairwise_results_10_mae_ratio_matrix.csv"
DEFAULT_OUTPUT_CSV = BASE_DIR / "perron_ranking_10.csv"


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Run Perron ranking on a pairwise ratio matrix.")
	parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV, help="Input ratio matrix CSV path")
	parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV, help="Output ranking CSV path")
	return parser.parse_args()


def perron_rank(adj_matrix: np.ndarray, max_iter: int = 100, tolerance: float = 1e-6) -> np.ndarray:
	n = adj_matrix.shape[0]
	r = np.ones(n) / n
	for _ in range(max_iter):
		new_r = np.dot(adj_matrix, r)
		new_r /= np.linalg.norm(new_r, 1)
		if np.linalg.norm(new_r - r) < tolerance:
			break
		r = new_r
	return r


def load_ratio_matrix(input_csv: Path) -> pd.DataFrame:
	if not input_csv.is_file():
		raise FileNotFoundError(f"Input CSV not found: {input_csv}")
	matrix = pd.read_csv(input_csv, index_col=0)
	matrix = matrix.apply(pd.to_numeric, errors="coerce").fillna(0.0)
	return matrix


def main() -> None:
	args = parse_args()
	ratio_matrix = load_ratio_matrix(args.input_csv)
	models = list(ratio_matrix.index)
	F = ratio_matrix.to_numpy(dtype=float)
	q = perron_rank(F)
	sorted_pairs = sorted(zip(models, q), key=lambda item: item[1], reverse=True)
	ranking_df = pd.DataFrame(sorted_pairs, columns=["model", "score"])
	args.output_csv.parent.mkdir(parents=True, exist_ok=True)
	ranking_df.to_csv(args.output_csv, index=False, encoding="utf-8-sig")
	print(sorted_pairs)
	print([pair[0] for pair in sorted_pairs])
	print(f"Saved ranking to {args.output_csv}")


if __name__ == "__main__":
	main()
