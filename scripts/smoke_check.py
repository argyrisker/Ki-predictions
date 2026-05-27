"""Fast project sanity check: data loading, RDKit parsing, and descriptor generation."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from apb_ki_modelling.data import clean_ki_values, load_dataset, standardise_and_aggregate
from apb_ki_modelling.descriptors import build_descriptor_matrices


def main() -> None:
    full_data = PROJECT_ROOT / "data" / "exam_dataset.csv"
    example_data = PROJECT_ROOT / "data" / "example_dataset.csv"
    data_path = full_data if full_data.exists() else example_data

    raw = load_dataset(data_path)
    cleaned = clean_ki_values(raw)
    agg, summary = standardise_and_aggregate(cleaned)
    sample_mols = agg["mol"].head(min(10, len(agg))).tolist()
    matrices = build_descriptor_matrices(sample_mols)

    print(f"Data file: {data_path}")
    print(summary)
    print("Sample descriptor matrix:", matrices["X"].shape)
    print("Smoke check passed.")


if __name__ == "__main__":
    main()
