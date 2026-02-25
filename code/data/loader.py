import pandas as pd
import numpy as np
import skrf as rf
from pathlib import Path
from tqdm import tqdm
from config import LEFT_LUNG_DIR, RIGHT_LUNG_DIR, REF_DIR


def load_s2p_directory(directory_path: Path, label: str) -> pd.DataFrame:
    """
    Reads all .s2p files in a directory and extracts S11 and S21 magnitudes.
    """
    if not directory_path.exists():
        raise FileNotFoundError(f"Directory missing: {directory_path}")

    files = list(directory_path.glob('*.s2p'))
    data_rows = []

    print(f"Processing {label} files...")
    for file_path in tqdm(files):
        # Read the Touchstone file using scikit-rf
        network = rf.Network(str(file_path))

        # network.f = frequencies, network.s = complex S-parameter matrix
        # network.s_db converts the matrix to magnitude in decibels
        # S11 is at index [:, 0, 0] and S21 is at [:, 1, 0]
        s11_db = network.s_db[:, 0, 0]
        s21_db = network.s_db[:, 1, 0]

        # Combine S11 and S21 into a single feature array
        features = np.concatenate([s11_db, s21_db])

        # Create a dictionary for this row
        row_data = {
            'filename': file_path.name,
            'label': label,
        }

        # Add the features as separate columns
        # (e.g., f_0_s11, f_1_s11 ... f_0_s21, f_1_s21 ...)
        for i, val in enumerate(features):
            row_data[f'feature_{i}'] = val

        data_rows.append(row_data)

    # Convert to DataFrame
    df = pd.DataFrame(data_rows)
    return df


def create_master_dataset() -> pd.DataFrame:
    """
    Compiles all three classes into a single master dataset.
    """
    df_left = load_s2p_directory(LEFT_LUNG_DIR, 'Left Tumor')
    df_right = load_s2p_directory(RIGHT_LUNG_DIR, 'Right Tumor')
    df_ref = load_s2p_directory(REF_DIR, 'Reference')

    master_df = pd.concat([df_left, df_right, df_ref], ignore_index=True)
    print(f"\nMaster Dataset created with shape: {master_df.shape}")
    return master_df