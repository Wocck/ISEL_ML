from pathlib import Path

from dataset import insert_data, orange_export_to_csv, r1_export_to_csv
from models import DatabaseConfig
from database_manager import DatabaseManager
from r1_model import OneRClassifier

db_config = DatabaseConfig(
    db_name="medknow",
    db_user="postgres",
    db_pass="postgres",
    db_host="172.31.17.248",
    db_port=5432
)

DATASET_SIZE = 300
BASE_DIR = Path(__file__).resolve().parent.parent

ORANGE_OUTPUT_FILE = BASE_DIR / "data" / "lenses_dataset.tab"
ORANGE_SQL_CREATE_DATASET_VIEW = BASE_DIR / "sql" / "export_orange.sql"
ORANGE_DATASET_TABLE = "orange_dataset"

R1_OUTPUT_FILE = BASE_DIR / "data" / "r1_dataset.tab"
R1_SQL_CREATE_DATASET_VIEW = BASE_DIR / "sql" / "export_r1.sql"
R1_DATASET_TABLE = "r1_dataset"


def run_r1_model():
    r1_model = OneRClassifier(R1_OUTPUT_FILE)
    r1_model.fit(target_col="lenses")
    print()
    print("===================================")
    print("========= OneRClassifier ==========")
    print("===================================")
    r1_model.pretty_print_rules()
    print("Accuracy:", round(r1_model.score() * 100, 2), "%")

def main():
    database_manager = DatabaseManager(config=db_config)
    
    # Creating and inserting synthetic data into EXAMINATION table
    insert_data(
        db=database_manager,
        records_num=DATASET_SIZE,
        delete_before_insert=True
    )
    
    # ORANGE DATAMINING: Creating export view in database and exporting it to .tab file
    orange_export_to_csv(
        db=database_manager,
        output_file=ORANGE_OUTPUT_FILE,
        sql_create_view_query=ORANGE_SQL_CREATE_DATASET_VIEW,
        dataset_table=ORANGE_DATASET_TABLE
    )
    
    # ONE-RULE MODEL: Creating export view in database and exporting it to .tab file
    r1_export_to_csv(
        db=database_manager,
        output_file=R1_OUTPUT_FILE,
        sql_create_view_query=R1_SQL_CREATE_DATASET_VIEW, 
        dataset_table=R1_DATASET_TABLE
    )
    
    # RUN ONE-RULE MODEL
    run_r1_model()
    
if __name__ == "__main__":
    main()
