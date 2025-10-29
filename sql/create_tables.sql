-- ==========================
-- ENUM type definitions
-- ==========================

CREATE TYPE tear_rate_type AS ENUM ('normal', 'reduced');
CREATE TYPE lenses_type AS ENUM ('hard', 'soft', 'none');
CREATE TYPE age_group_type AS ENUM ('young', 'pre-presbyopic', 'presbyopic');
CREATE TYPE disease_type AS ENUM ('myope', 'hypermetrope', 'astigmatic');

-- ==========================
-- Table definitions
-- ==========================

CREATE TABLE PATIENT (
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

