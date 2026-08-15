"""
Convert checked-in CSV data files to the .npy layout used by the refactored code.
"""

import os

import numpy as np


CSV_TO_NPY = {
    "training/training_100.csv": "data/training/training_100.npy",
    "given_truths.csv": "data/truth/given_truths.npy",
    "desired_truths.csv": "data/truth/desired_truths.npy",
}


def convert_csv_to_npy(csv_path, npy_path):
    values = np.genfromtxt(csv_path, delimiter=",", skip_header=1)
    os.makedirs(os.path.dirname(npy_path), exist_ok=True)
    np.save(npy_path, values)
    print(f"Saved {npy_path}")


def main():
    for csv_path, npy_path in CSV_TO_NPY.items():
        convert_csv_to_npy(csv_path, npy_path)


if __name__ == "__main__":
    main()
