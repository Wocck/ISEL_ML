from pathlib import Path

from dataset import insert_data, orange_export_to_csv, export_to_csv
from models import DatabaseConfig
from database_manager import DatabaseManager

from r1_model import OneRClassifier
from id3_model import ID3Classifier
from naive_bayes_model import NaiveBayesClassifier

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

TAB_DATASET_FILE = BASE_DIR / "data" / "dataset.tab"
MODELS_SQL_CREATE_DATASET_VIEW = BASE_DIR / "sql" / "export_models.sql"
MODELS_DATASET_TABLE = "models_dataset"


def run_r1_model():
    r1_model = OneRClassifier(TAB_DATASET_FILE)
    r1_model.fit(target_col="lenses")
    print()
    print("=====================================")
    print("======== One Rule Classifier ========")
    print("=====================================")
    r1_model.pretty_print_rules()
    print("Accuracy:", round(r1_model.score() * 100, 2), "%")

def run_id3_model():
    id3_model = ID3Classifier(TAB_DATASET_FILE)
    id3_model.fit("lenses")
    print()
    print("======================================")
    print("========== ID3 Classifier ============")
    print("======================================")
    print("Accuracy:", round(id3_model.score() * 100, 2), "%")
    id3_model.print_tree()

def run_naive_bayes_model():
    naive_bayes_model = NaiveBayesClassifier(TAB_DATASET_FILE)
    naive_bayes_model.fit("lenses")
    print()
    print("======================================")
    print("======= Naive Bayes Classifier =======")
    print("======================================")
    print("Accuracy:", round(naive_bayes_model.score() * 100, 2), "%")
    

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
    
    # MODEL: Creating export view in database and exporting it to .tab file
    export_to_csv(
        db=database_manager,
        output_file=TAB_DATASET_FILE,
        sql_create_view_query=MODELS_SQL_CREATE_DATASET_VIEW, 
        dataset_table=MODELS_DATASET_TABLE
    )
    
    # RUN ONE-RULE MODEL
    run_r1_model()
    # RUN ID3 MODEL
    run_id3_model()
    # RUN NAIVE BAYES MODEL
    run_naive_bayes_model()


if __name__ == "__main__":
    main()
