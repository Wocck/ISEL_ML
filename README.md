# Project A

This project implements a classification system that predicts the type of contact lenses a patient should receive based on ophthalmological examination data.
The models are trained on synthetic but medically consistent data, generated directly into a PostgreSQL database.

Three machine learning models are implemented from scratch:  
- `One Rule (1R)`  
- `ID3 Decision Tree`  
- `Naive Bayes Classifier`  

The project also includes:  
- Export of data for Orange Data Mining  
- Model evaluation with train/test split and 5-fold cross validation  
- A web interface (FastAPI) for testing predictions interactively  

## Structure

```bash
Project_A/
├── data/                      # Generated datasets (.tab)
├── docs/                      # Final report + documentation
├── graphics/                  # ER diagram + database diagram
├── scripts/                   # Python implementation of models + main runner
│   ├── config.py              # Path, Database, execution parameters configuration
│   ├── main.py                # Generates data, trains and evaluates models
│   ├── r1_model.py            # One Rule classifier
│   ├── id3_model.py           # ID3 Decision Tree classifier
│   ├── naive_bayes_model.py   # Naive Bayes classifier
│   ├── dataset.py             # Synthetic data generation and exports
│   ├── evaluation.py          # Test accuracy + cross-validation utilities
│   └── database_manager.py    # Database connector
├── sql/                       # Schema creation & dataset export queries
└── webapp/                    # FastAPI UI for predictions
    ├── app.py
    └── templates/index.html

```

## Requirements

- `Python 3.11+`
- `PostgreSQL` - running locally or remotely (connection settings configured in scripts/config.py)

**Install required dependencies:**
1. With `pip`:
```bash
python -m venv .venv
.venv\Scripts\activate
pip install fastapi uvicorn jinja2 pandas psycopg2
```

2. With `uv` (Recomended):  

**UV instalation:**
```
# On macOS and Linux.
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows.
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# With pip.
pip install uv
```  

**Dependencies Instalation:**
```
uv venv
uv sync
```  

## Running project

### 1. Configure Databse

Configure PostgreSQL database connection with `DB_CONFIG` variable in `scripts/config.py`:
```python
DB_CONFIG = DatabaseConfig(
    db_name="Yours Database Name",
    db_user="Yours User Name",
    db_pass="Yours User Password",
    db_host="Host",
    db_port=5432    # default Postgres port
)
```

### 2. Generate Data Train and evaluate Models

This will:
- Populate the database with sample records  
- Export datasets .tab  
- Train and evaluate all models (1R, ID3, Naive Bayes)  
- Print rules / tree structure and accuracy results  

```bash
.venv\Scripts\activate
python scripts/main.py
```

### 3. Models Deployment

This will:
- Launch a simple web interface for testing model predictions  
- Allow entering patient attributes and selecting classifier  


```bash
.venv\Scripts\activate
uvicorn webapp.app:app --reload
```

Webapp will be available under ` http://127.0.0.1:8000`