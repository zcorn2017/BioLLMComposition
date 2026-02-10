import os
from pathlib import Path
from dotenv import load_dotenv

REPO_DIR = Path(__file__).resolve().parents[1]
load_dotenv(REPO_DIR / ".env", override=False)

# external data lake
DATA_DIR = Path(os.getenv("DATA_DIR", "/home/zcorn/Projects/proteinDNA_data")).expanduser().resolve()

# repo-local small data (optional)
LOCAL_DATA = (REPO_DIR / "data").resolve()
TRAIN_TSV = LOCAL_DATA / "combo_1and2_train.tsv"
VALID_TSV = LOCAL_DATA / "combo_1and2_valid.tsv"
