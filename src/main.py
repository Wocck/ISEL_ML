import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

from dataset import insert_data, orange_export_to_csv, export_to_csv
from evaluation import summarize_cv
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
    r1_model = OneRClassifier()
    r1_model.set_training_data(train_df)
    r1_model.fit(target_col="lenses")
    r1_model.run_evaluate(test_df=test_df)

def run_id3_model():
    id3_model = ID3Classifier()
    id3_model.set_training_data(train_df)
    id3_model.fit("lenses")
    id3_model.run_evaluate(test_df=test_df)

def run_naive_bayes_model():
    nb = NaiveBayesClassifier()
    nb.set_training_data(train_df)
    nb.fit("lenses")
    nb.run_evaluate(test_df=test_df)

def main():
    global train_df, test_df 
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
    
    # LOAD AND SPLIT DATA
    df = pd.read_csv(TAB_DATASET_FILE, sep="\t")
    train_df, test_df = train_test_split(df, test_size=0.2, shuffle=True, random_state=42)
    
    # RUN ONE-RULE MODEL
    run_r1_model()
    # RUN ID3 MODEL
    run_id3_model()
    # RUN NAIVE BAYES MODEL
    run_naive_bayes_model()
    
    # K-Fold Cross-Validation
    summarize_cv(df, target_col="lenses", k=5)


if __name__ == "__main__":
    main()