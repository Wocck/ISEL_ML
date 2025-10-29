# Machine Learning and Data Mining - FINAL PROJECT A

**Date:** *29.10.2025*  
**Working Group:** *D16*  
**Authors:**  
| Name              | Student Number    |
|-------------------|-------------------|
| Wojciech Sekula   | A54219            |
| Keira Barazzoli   | A54213            |
| Jakub Wilk        | A53995            |
| Sipos Máté        | A54216            |

---

## 1. Introduction

### 1.1 Problem Context
The company MedKnow is a medical center that collects data about patients during ophthalmological examinations. Each patient is examined by a doctor and may present conditions such as myopia, hypermetropia, or astigmatism. The goal of the system is to provide **support for lens prescription** by learning from historical data and identifying patterns that indicate the most appropriate lens type (soft, hard, or none).

### 1.2 Data Understanding
The dataset includes the following descriptive features:
- **Age group:** {young, pre-presbyopic, presbyopic}  
- **Disease name:** {myope, hypermetrope, astigmatic}  
- **Astigmatism:** {yes, no}  
- **Tear production:** {normal, reduced}  
The **target attribute** is: **lenses** ∈ {soft, hard, none}

This dataset is small and categorical, which makes it suitable for **rule-based and tree-based classifiers**.

## 2. Conceptual and Logical Data Model

### 2.1 Conceptual Model (ER Diagram)

![ER Diagram](../graphics/ER_diagram.png)

The system considers the following main entities:
- **PATIENT**
- **DOCTOR**
- **DISEASE**
- **EXAMINATION** (link between Patient and Doctor, storing condition and prescribed lens)

### 2.2 Logical Schema and Implementation
The database is implemented in PostgreSQL and includes primary keys, foreign keys, and controlled vocabulary constraints.  

```sql
CREATE TYPE tear_rate_type AS ENUM ('normal', 'reduced');
CREATE TYPE lenses_type AS ENUM ('hard', 'soft', 'none');
CREATE TYPE age_group_type AS ENUM ('young', 'pre-presbyopic', 'presbyopic');
CREATE TYPE disease_type AS ENUM ('myope', 'hypermetrope', 'astigmatic');

REATE TABLE PATIENT (
    patient_id SERIAL PRIMARY KEY,
    name VARCHAR(50),
    birth_date DATE,
    age_group age_group_type NOT NULL
);

CREATE TABLE DOCTOR (
    doctor_id SERIAL PRIMARY KEY,
    name VARCHAR(50),
    specialization VARCHAR(50)
);

CREATE TABLE DISEASE (
    disease_id SERIAL PRIMARY KEY,
    disease_name disease_type NOT NULL
);

CREATE TABLE EXAMINATION (
    exam_id SERIAL PRIMARY KEY,
    exam_date DATE,
    astigmatic BOOLEAN,
    tear_rate tear_rate_type NOT NULL,
    lenses lenses_type NOT NULL,
    patient_id INT REFERENCES PATIENT(patient_id),
    doctor_id INT REFERENCES DOCTOR(doctor_id),
    disease_id INT REFERENCES DISEASE(disease_id)
);
```

![Database Diagram](../graphics/db_diagram.png)

---

## 3. Project

### 3.1 Project Structure

```bash
[].
├── [] data         # Input data for models and Orange Datamining.
├── [] docs         # Project documentation.
├── [] graphics     # Images with diagrams.
├── [] orange       # .ows files for Orange Datamining.
├── [] scripts      # Python source code of models, dataset export scripts,
                    # data models and demo main() execution.
├── [] sql          # SQL scripts for creating tables and views.
├── [] webapp       # Source code for Web application interface.
└── X README.md     # file with deploment instructions.
```

### 3.2 Specific files description

**src:**

- `dataset.py` - contains 3 functions for creating synthetic lenses data, exporting data to *.tsv * file for Orange datamining software and exporting to *.tsv* file for 1R, ID3 and Naive Bayes models. When generating data function follows these rules so that the output is realistic:
```python
    if tear == TearRate.REDUCED:
        return LensType.NONE
    if disease == Disease.MYOPE and astig:
        return LensType.HARD
    if disease == Disease.HYPERMETROPE and tear == TearRate.NORMAL:
        return LensType.SOFT
    return LensType.SOFT
```
- `models.py` - contains dataclass definition for Database config variable and Enums for database values.
- `r1_model.py`, `src/id3_model.py`, `src/naive_bayes_model.py` - implementations of machine learning models
- `main.py` - contains execution of creating and exporting data for models and Orange software usage, it also runs every model against the dataset and prints their metrics (accuracy, error rate etc.)

**sql:**
- `create_tables.sql` - query for creating tables and enums in database
- `export_models.sql` - query for exporting EXAMINATIONS data to *.tsv* file 
- `export_orange.sql` - query for exporting EXAMINATIONS data to *.tsv* file in *Orange Datamining* format

**others:**
- `webapp/` - contains web aplication with very simple html template for testing models. It runs on FastAPI and can return prediction for given set of input attributes.
- `README.md` - instructions for runing main script and deploying *webapp*

### 3.3 Classification Methods Implemented

#### 3.3.1 One Rule (1R)

`OneR (One Rule)` is a very simple classification algorithm that builds just one rule based on the dataset.
For each attribute, the algorithm checks which class appears most frequently for every value of that attribute, and creates a rule assigning that class. Then it calculates the number of classification errors for that attribute. After evaluating all attributes, `OneR` chooses the attribute with the lowest error rate and uses it as the final rule.  
Even though OneR is extremely simple, it often achieves surprisingly good accuracy, while keeping the resulting model easy to understand and explain, which makes it useful for decision support systems.  

**Algorithm**:
```
For each predictor
    For each value of that predictor, make a rule as follows
        1. Count how often each value of target (class) appears
        2. Find the most frequent class
        3. Make the rule assign that class to this value of the predictor
    Calculate the total error / accuracy of the rules of each predictor
Choose the predictor with the smallest total error or highest accuracy.
```

#### 3.3.2 ID3 Decision Tree

ID3 stands for Iterative Dichotomiser 3 and is named such because the algorithm repeatedly divides features into two or more groups at each step creating a decision tree. A decision tree is a classification model that makes predictions by recursively splitting the data into branches based on attribute values, forming a tree-like structure of decisions and outcomes. ID3 uses a top-down greedy approach to build a decision tree that means we start from the root and select the best feature at the present moment of execution.

ID3 uses Information Gain or just Gain to find the best feature. Information Gain calculates the reduction in the entropy and measures how well a given feature separates or classifies the target classes. Entropy is the measure of disorder and the Entropy of a dataset is the measure of disorder in the target feature of the dataset.  
```Entropy(S) = - ∑ pᵢ * log₂(pᵢ) ; i = 1 to n```
where:  
- `n` is the total number of classes in the target column,
- `pᵢ` is the fraction of records that belong to class `i`. We calculate it by dividing the number of rows with class `i` by the total number of rows in the dataset. `pᵢ` tells us how common each class is in the data.

Information gain is calculated as:
```IG(S, A) = Entropy(S) - ∑((|Sᵥ| / |S|) * Entropy(Sᵥ))```
where:  
- `Sᵥ` is the set of rows in `S` for which the feature column `A` has value `v`, 
- `|Sᵥ|` is the number of rows in `Sᵥ`,
- `|S|` is the number of rows in `S`

**Steps to build decision tree:**  
1. Calculate the Information Gain for each feature in the dataset.
2. Select the feature with the highest Information Gain, because it separates the classes best.
3. Create a decision tree node using this feature and split the dataset into subsets based on its values.
4. If a subset contains only one class, make it a leaf node labeled with that class.
5. Repeat the process for each subset using the remaining features, until:
    - there are no more features to split on, or
    - all nodes become leaf nodes.

To make a prediction with a decision tree, we start at the root node and follow the branches based on the values of the input features. At each internal node, we check the feature used for the split and choose the branch that matches the input. We continue moving down the tree until we reach a leaf node, whose label is the final predicted class.

#### 3.3.3 Naive Bayes Classifier



---

## 4. Conclusions

### 4.1 Model Evaluation and Comparison
