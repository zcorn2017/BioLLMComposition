import os
from pathlib import Path


from dotenv import load_dotenv




# Get the directory where the script or notebook lies
PROJECT_DIR = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_DIR)
load_dotenv()
# Add 'data' directory path (as a variable) so it can access /home/zcorn/Projects/proteinDNA_data
DATA_DIR = Path(os.getenv("DATA_DIR", "/home/zcorn/Projects/proteinDNA_data")).resolve()


