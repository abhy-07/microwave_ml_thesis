from pathlib import Path

# Base and Data directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'Datafiles'

# Specific Lungs Data directories
LUNGS_DATA_DIR = DATA_DIR / 'Lungs Microwave Images new'
LEFT_LUNG_DIR = LUNGS_DATA_DIR / 'Left Lung 30'
RIGHT_LUNG_DIR = LUNGS_DATA_DIR / 'Right lung 30'
REF_DIR = LUNGS_DATA_DIR / 'Ref'

# Output Directory Configuration
OUTPUT_DIR = BASE_DIR / 'code' / 'output'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    directories = [LEFT_LUNG_DIR, RIGHT_LUNG_DIR, REF_DIR]
    for d in directories:
        if d.exists():
            file_count = len(list(d.glob('*.s2p')))
            print(f"Found {d.name}: {file_count} .s2p files")
        else:
            print(f"WARNING: Directory not found -> {d}")

    print(f"Outputs will be saved to: {OUTPUT_DIR}")