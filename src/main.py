import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

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

df = pd.read_csv(TAB_DATASET_FILE, sep="\t")
train_df, test_df = train_test_split(df, test_size=0.2, shuffle=True, random_state=42)


def run_r1_model():
    r1_model = OneRClassifier()
    r1_model.set_training_data(test_df)
    r1_model.fit(target_col="lenses")
    print()
    print("=====================================")
    print("======== One Rule Classifier ========")
    print("=====================================")
    r1_model.pretty_print_rules()
    print("Train Accuracy:", round(r1_model.score() * 100, 2), "%")
    test_acc = evaluate_on_test(r1_model, test_df, "lenses")
    print("Test Accuracy:", round(test_acc * 100, 2), "%")

def run_id3_model():
    id3_model = ID3Classifier()
    id3_model.set_training_data(test_df)
    id3_model.fit("lenses")
    print()
    print("======================================")
    print("========== ID3 Classifier ============")
    print("======================================")
    print("Train Accuracy:", round(id3_model.score() * 100, 2), "%")
    test_acc = evaluate_on_test(id3_model, test_df, "lenses")
    print("Test Accuracy:", round(test_acc * 100, 2), "%")
    id3_model.print_tree()

def run_naive_bayes_model():
    naive_bayes_model = NaiveBayesClassifier()
    naive_bayes_model.set_training_data(test_df)
    naive_bayes_model.fit("lenses")
    print()
    print("======================================")
    print("======= Naive Bayes Classifier =======")
    print("======================================")
    print("Train Accuracy:", round(naive_bayes_model.score() * 100, 2), "%")
    test_acc = evaluate_on_test(naive_bayes_model, test_df, "lenses")
    print("Test Accuracy:", round(test_acc * 100, 2), "%")

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


def evaluate_on_test(model, test_df, target_col):
    test_df = test_df.copy()

    # norm as string
    for c in test_df.columns:
        test_df[c] = test_df[c].astype(str).str.strip()

    preds = []
    real = test_df[target_col].tolist()

    for i in range(len(test_df)):
        preds.append(model.predict_row(test_df.iloc[i]))

    correct = sum(1 for p, r in zip(preds, real) if p == r)
    return correct / len(real)
