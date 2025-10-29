# Project A

## Structure

```bash
Project_A/
├── data/                      # Exported datasets (.tab)
├── docs/                      # Additional documentation
├── graphics/                  # ER diagram (draw.io + PNG)
├── scripts/                   # Database logic and ML model implementations
├── sql/                       # Schema creation, population, export views
└── webapp/                    # FastAPI UI for model predictions
```

## Key components

- `sql/*.sql`                   Database schema, constraints, population, dataset export
- `scripts/main.py`             Generates dataset and trains the 1R model
- `scripts/r1_model.py`         Custom implementation of the 1R classifier
- `webapp/app.py`               Minimal FastAPI backend with prediction endpoint
- `webapp/templates/index.html` Simple HTML form interface

## Requirements

- `Python 3.11+`
- `PostgreSQL` (database connection configured in scripts/main.py)

**Install required dependencies:**
1. With `pip`:
```bash
python -m venv .venv
.venv\Scripts\activate
pip install fastapi uvicorn jinja2 pandas tqdm psycopg2
```

2. With `uv`:
```
uv venv
uv sync
```

## Running project

1. **Generate Data and Train Model**
This will:
- Populate the database with sample records
- Export datasets .tab
- Train and display the 1R model + accuracy

```bash
.venv\Scripts\activate
python scripts/main.py
```

2. **Start Web Application**
```bash
.venv\Scripts\activate
uvicorn webapp.app:app --reload
```