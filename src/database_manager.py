import psycopg2
from pathlib import Path
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager
from psycopg2 import OperationalError, InterfaceError, DatabaseError, ProgrammingError, IntegrityError

from src.models import DatabaseConfig
from src.config import DROP_CREATE_TABLES, INITIAL_POPULATE_DATABASE
class DatabaseManager:
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.conn = None
        self.cur = None
        self.connect()
        self.execute_sql_file(DROP_CREATE_TABLES)
        self.execute_sql_file(INITIAL_POPULATE_DATABASE)
        print("[OK] Database connected")

    def connect(self):
        try:
            self.conn = psycopg2.connect(
                dbname=self.config.db_name,
                user=self.config.db_user,
                password=self.config.db_pass,
                host=self.config.db_host,
                port=self.config.db_port
            )
            self.cur = self.conn.cursor(cursor_factory=RealDictCursor)
        except Exception as e:
            print(self._human_message_from_exception(e))

    def close(self):
        if self.cur:
            self.cur.close()
        if self.conn:
            self.conn.close()

    def execute_sql_file(self, sql_path: Path):
        try:
            with open(sql_path, "r", encoding="utf-8") as f:
                sql_commands = f.read()
                self.execute(sql_commands, commit=True)
        except Exception as e:
            print(self._human_message_from_exception(e))

    @contextmanager
    def session(self):
        try:
            self.connect()
            yield self
        finally:
            self.close()

    def execute(self, query: str, params=None, commit=False):
        if not self.cur:
            print("Database cursor is not initialized. Call connect() first.")
            return
        if not self.conn:
            print("Database connection is not initialized. Call connect() first.")
            return
        try:
            self.cur.execute(query, params)
        except Exception as e:
            print(self._human_message_from_exception(e))
        if commit:
            self.conn.commit()

    def fetch_all(self, query: str, params=None):
        if not self.cur:
            print("Database cursor is not initialized. Call connect() first.")
            return
        try:
            self.cur.execute(query, params)
        except Exception as e:
            print(self._human_message_from_exception(e))
        return self.cur.fetchall()

    def fetch_one(self, query: str, params=None):
        if not self.cur:
            print("Database cursor is not initialized. Call connect() first.")
            return
        try:
            self.cur.execute(query, params)
        except Exception as e:
            print(self._human_message_from_exception(e))
        return self.cur.fetchone()

    def get_patients(self):
        rows = self.fetch_all("SELECT patient_id FROM PATIENT;")
        if rows:
            return [r["patient_id"] for r in rows]
        else:
            return []

    def get_doctors(self):
        rows = self.fetch_all("SELECT doctor_id FROM DOCTOR;")
        if rows:
            return [r["doctor_id"] for r in rows]
        else:
            return []

    def get_disease_map(self):
        rows = self.fetch_all("SELECT disease_id, disease_name FROM DISEASE;")
        if rows:
            return {r["disease_name"]: r["disease_id"] for r in rows}
        else:
            return {}

    def get_patient_age_group(self, patient_id):
        result = self.fetch_one("SELECT age_group FROM PATIENT WHERE patient_id=%s;", (patient_id,))
        return result["age_group"] if result else None

    def clear_examinations(self):
        self.execute("DELETE FROM examination;", commit=True)

    def insert_examination_record(self, exam_date, astig, tear, lens, patient, doctor, disease_id):
        try:
            self.execute("""
                INSERT INTO examination (exam_date, astigmatic, tear_rate, lenses, patient_id, doctor_id, disease_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s);
            """, (exam_date, astig, tear, lens, patient, doctor, disease_id))
        except Exception as e:
            print(self._human_message_from_exception(e))

    def get_dataset_rows(self, dataset_table: str):
        return self.fetch_all(f"SELECT * FROM {dataset_table};")
    
    def _human_message_from_exception(self, exc: Exception) -> str:
        if isinstance(exc, OperationalError):
            msg = str(exc).strip()
            if "authentication failed" in msg.lower() or "password authentication failed" in msg.lower() or "28p01" in msg.lower():
                return "Authentication failed: check username/password or acces rights to database."
            if "could not connect to server" in msg.lower() or "connection refused" in msg.lower():
                return "Cannot connect to database server (host/port)."
            if "timeout expired" in msg.lower() or "timeout" in msg.lower():
                return "Connection Timeout."
            return f"OperationalError: {msg}"
        elif isinstance(exc, InterfaceError):
            return f"Database interface error: {str(exc)}"
        elif isinstance(exc, IntegrityError):
            return f"Constraint violation. details: {str(exc)}"
        elif isinstance(exc, ProgrammingError):
            return f"SQL query error: {str(exc)}"
        elif isinstance(exc, DatabaseError):
            return f"Database error: {str(exc)}"
        else:
            return str(exc)

