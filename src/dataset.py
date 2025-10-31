# ============================================================
# Generating and extracting synthetic examination data
# ============================================================
# Decision Logic Used to Assign Lens Type:
#   1) tear_rate = reduced
#       lenses = none
#   2) disease = myope AND astigmatic = true
#       lenses = hard
#   3) disease = hypermetrope AND tear_rate = normal
#       lenses = soft
#   4) default case
#       lenses = soft
# ============================================================

import random
import csv
import pandas as pd
from datetime import date, timedelta
from pathlib import Path


from models import Disease, TearRate, LensType
from database_manager import DatabaseManager



def choose_lens(disease: Disease, astig: bool, tear: TearRate) -> LensType:
    if tear == TearRate.REDUCED:
        return LensType.NONE
    if disease == Disease.MYOPE and astig:
        return LensType.HARD
    if disease == Disease.HYPERMETROPE and tear == TearRate.NORMAL:
        return LensType.SOFT
    return LensType.SOFT


def insert_data(db: DatabaseManager, records_num: int, delete_before_insert: bool):
    try:
        print("[INFO] Creating data ...")
        patients = db.get_patients()
        doctors = db.get_doctors()
        disease_map = db.get_disease_map()

        base_date = date(2025, 1, 1)
        
        if delete_before_insert:
            db.clear_examinations()
        
        for i in range(records_num):
            patient = random.choice(patients)
            doctor = random.choice(doctors)
            disease = random.choice(list(Disease))
            if disease == Disease.ASTIGMATIC:
                astigmatic = True
            else:
                astigmatic = random.random() < 0.3
            tear = random.choice(list(TearRate))
            
            lens = choose_lens(disease, astigmatic, tear)
            disease_id = disease_map[disease.value]
            exam_date = base_date + timedelta(days=i)

            db.insert_examination_record(
                exam_date,
                astigmatic,
                tear.value,
                lens.value,
                patient,
                doctor,
                disease_id
            )
        
        if db.conn:
            db.conn.commit()
        print(f"[OK] Generated {records_num} records for table examination.")

    except Exception as e:
        print("[Error]", e)


def orange_export_to_csv(db: DatabaseManager, output_file: Path, sql_create_view_query: Path, dataset_table: str):
    try:
        print("[INFO] Updating database views before data export")
        db.refresh_views(sql_create_view_query)
        print("[OK] Views refreshed successfully.")
        
        rows = db.get_dataset_rows(dataset_table)
        if not rows:
            raise RuntimeError("Query returned no data. Check the view definition.")
        
        with open(output_file, "w", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            for row in rows:
                writer.writerow(row.values())

        print(f"[OK] Dataset for Orange DataMining exported to: {output_file}")

    except Exception as e:
        print("[ERROR] Something went wrong:")
        print(e)

def export_to_csv(db: DatabaseManager, sql_create_view_query: Path, dataset_table: str, output_file: Path) -> Path:
    db.refresh_views(sql_create_view_query)
    rows = db.get_dataset_rows(dataset_table)

    if not rows:
        raise RuntimeError("No data in view dataset_table.")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame(rows)
    df.to_csv(output_file, sep="\t", index=False)

    print(f"[OK] Dataset for model training exported to: {output_file}")
    return output_file
