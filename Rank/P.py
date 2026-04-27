from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = BASE_DIR / "pairwise_results_10_mae"
DEFAULT_OUTPUT_PDF = BASE_DIR / "pairwise_results_10_mae_heatmap.pdf"
DEFAULT_OUTPUT_CSV = BASE_DIR / "pairwise_results_10_mae_matrix.csv"
MODEL_DISPLAY_NAMES = {
	"cflow-ad": "CFLOW-AD",
	"destseg": "DeSTSeg",
	"dsr": "DSR",
	"rd": "RD",
	"urd": "URD",
	"invad": "InvAD",
	"uninet": "UniNet",
	"deco-diff": "DeCo-Diff",
	"comet": "CoMet-SN",
	"rrd": "RD++",
	"patchcore": "PatchCore",
	"dinomaly": "Dinomaly",
	"simplenet": "SimpleNet",
	"msflow": "MSFlow",
}
MODEL_ORDER = [
	"cflow-ad",
	"destseg",
	"dsr",
	"rd",
	"urd",
	"invad",
	"uninet",
	"deco-diff",
	"comet",
	"rrd",
	"patchcore",
	"dinomaly",
	"simplenet",
	"msflow",
]


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Build a pairwise MAE heatmap from summary/overall rows in pairwise_results_1_mae."
	)
	parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR, help="Directory with pairwise xlsx files")
	parser.add_argument("--output-pdf", type=Path, default=DEFAULT_OUTPUT_PDF, help="Output heatmap PDF path")
	parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV, help="Output MAE matrix CSV path")
	return parser.parse_args()


def collect_workbooks(input_dir: Path) -> list[Path]:
	if not input_dir.is_dir():
		raise FileNotFoundError(f"Input directory not found: {input_dir}")
	return sorted(
		path
		for path in input_dir.glob("*.xlsx")
		if path.is_file() and not path.name.startswith("~$")
	)


def parse_model_pair(workbook_path: Path) -> tuple[str, str]:
	name = workbook_path.stem
	if "_top_" in name:
		name = name.split("_top_", 1)[0]
	if "_vs_" not in name:
		raise ValueError(f"Cannot parse model pair from filename: {workbook_path.name}")
	left_model, right_model = name.split("_vs_", 1)
	if left_model not in MODEL_DISPLAY_NAMES:
		raise ValueError(f"Unsupported model name in {workbook_path.name}: {left_model}")
	if right_model not in MODEL_DISPLAY_NAMES:
		raise ValueError(f"Unsupported model name in {workbook_path.name}: {right_model}")
	return left_model, right_model


def load_overall_mae_matrix(input_dir: Path) -> pd.DataFrame:
	workbooks = collect_workbooks(input_dir)
	if not workbooks:
		raise ValueError(f"No xlsx files found in {input_dir}")

	models: set[str] = set()
	for workbook_path in workbooks:
		left_model, right_model = parse_model_pair(workbook_path)
		models.add(left_model)
		models.add(right_model)

	model_list = [model for model in MODEL_ORDER if model in models]
	if len(model_list) != len(models):
		missing_models = sorted(models.difference(MODEL_ORDER))
		raise ValueError(f"Unexpected models found in input directory: {missing_models}")
	matrix = pd.DataFrame(index=model_list, columns=model_list, dtype=float)

	for workbook_path in workbooks:
		left_model, right_model = parse_model_pair(workbook_path)
		summary = pd.read_excel(workbook_path, sheet_name="summary")
		type_values = summary["type"].astype(str).str.strip().str.lower()
		name_values = summary["name"].astype(str).str.strip().str.lower()
		overall_rows = summary[
			(
				(type_values == "overall")
				| (type_values == "overall_model")
				| name_values.eq("overall")
			)
			& name_values.eq("overall")
		]
		if overall_rows.empty:
			raise ValueError(
				f"overall row not found in {workbook_path}. "
				f"Available rows: {summary[['type', 'name']].to_dict('records')}"
			)
		overall = overall_rows.iloc[0]

		left_col = f"{left_model}_mae"
		right_col = f"{right_model}_mae"
		if left_col not in overall or right_col not in overall:
			raise ValueError(f"Missing MAE columns in {workbook_path.name}")

		matrix.loc[left_model, right_model] = float(overall[left_col])
		matrix.loc[right_model, left_model] = float(overall[right_col])

	for model in matrix.index:
		matrix.loc[model, model] = 0.0

	matrix = matrix.rename(index=MODEL_DISPLAY_NAMES, columns=MODEL_DISPLAY_NAMES)
	return matrix


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
	green_cmap = sns.light_palette("#2e7d32", as_cmap=True)
	sns.heatmap(
		matrix,
		ax=ax,
		cmap=green_cmap,
		annot=True,
		fmt=".3f",
		linewidths=0.5,
		linecolor="#d7ead1",
		mask=matrix.isna(),
		annot_kws={"size": 16, "family": "Times New Roman", "color": "black"},
		#cbar_kws={"label": "MAE"},
	)
	#ax.set_title("Pairwise Overall MAE Heatmap", pad=16, color="#245c27")
	#ax.set_xlabel("Opponent model")
	#ax.set_ylabel("Target model")
	plt.xticks(rotation=45, ha="right")
	plt.yticks(rotation=0)
	ax.tick_params(axis="both", labelsize=15)
	colorbar = ax.collections[0].colorbar
	colorbar.ax.tick_params(labelsize=16)
	#colorbar.set_label("MAE", size=16)
	fig.tight_layout()
	fig.savefig(output_pdf, format="pdf", bbox_inches="tight")
	plt.close(fig)


def main() -> None:
	args = parse_args()
	matrix = load_overall_mae_matrix(args.input_dir)
	matrix = matrix.round(3)
	matrix.to_csv(args.output_csv, encoding="utf-8-sig", float_format="%.3f")
	save_heatmap(matrix, args.output_pdf)
	print(f"Saved MAE matrix to {args.output_csv}")
	print(f"Saved heatmap to {args.output_pdf}")
	print(matrix.to_string(float_format=lambda value: f"{value:.3f}"))


if __name__ == "__main__":
	main()
