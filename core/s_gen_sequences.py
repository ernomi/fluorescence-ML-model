#!/usr/bin/env python3
"""
core/s_gen_sequences.py

Generates a synthetic CSV file of fluorescence-related data.
Usage example:
    python core/s_gen_sequences.py --num_samples 500 --out_file s_data.csv
"""
import argparse
import csv
import numpy as np
import sys
from pathlib import Path


def generate_sequence(num_samples: int = 500, out_file: str = "s_data.csv", seed: int | None = None):
    """
    Generate a synthetic CSV dataset for ML training.

    Args:
        num_samples (int): Number of rows/samples to generate.
        out_file (str): Path to output CSV file.
        seed (int, optional): Random seed for reproducibility.
    """
    if seed is not None:
        np.random.seed(seed)

    out_path = Path(out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating synthetic data: {num_samples} samples → {out_path}")
    sys.stdout.flush()

    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)

        # Write header
        writer.writerow(["Sample_ID", "initial_amount", "sample_concentration", "spore_count"])

        # Generate random data
        for i in range(num_samples):
            sample_id = i
            initial_amount = np.random.randint(98, 102)
            sample_concentration = np.random.randint(1, 5)

            # Conditional fluorescence → affects spore_count
            if sample_concentration > 2:
                spore_count = np.random.randint(0, 20)
            else:
                spore_count = np.random.randint(0, 200)

            writer.writerow([sample_id, initial_amount, sample_concentration, spore_count])

            if (i + 1) % max(1, num_samples // 10) == 0:
                print(f"  Progress: {i+1}/{num_samples}")
                sys.stdout.flush()

    print("CSV generation complete.")
    return {"output_file": str(out_path), "num_samples": num_samples}


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic fluorescence sample CSV data.")
    parser.add_argument("--num_samples", type=int, default=500, help="Number of samples to generate")
    parser.add_argument("--out_file", type=str, default="s_data.csv", help="Output CSV filename")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (optional)")
    args = parser.parse_args()

    generate_sequence(num_samples=args.num_samples, out_file=args.out_file, seed=args.seed)


if __name__ == "__main__":
    main()
