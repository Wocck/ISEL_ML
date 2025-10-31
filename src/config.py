from pathlib import Path

from models import DatabaseConfig

# Database connection config
DB_CONFIG = DatabaseConfig(
    db_name="medknow",
    db_user="postgres",
    db_pass="postgres",
    db_host="172.31.17.248",
    db_port=5432
)

# Size of generated dataset
DATASET_SIZE = 300

BASE_DIR = Path(__file__).resolve().parent.parent

ORANGE_OUTPUT_FILE = BASE_DIR / "data" / "lenses_dataset.tab"
ORANGE_SQL_CREATE_DATASET_VIEW = BASE_DIR / "sql" / "export_orange.sql"
# Name of the view in database
ORANGE_DATASET_TABLE = "orange_dataset"

TAB_DATASET_FILE = BASE_DIR / "data" / "dataset.tab"
MODELS_SQL_CREATE_DATASET_VIEW = BASE_DIR / "sql" / "export_models.sql"
# Name of the view in database
MODELS_DATASET_TABLE = "models_dataset"
