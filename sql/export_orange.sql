-- ==================================================
-- ORANGE .TAB EXPORT VIEW
-- ==================================================

-- Row 1: Column names
CREATE OR REPLACE VIEW orange_header_names AS
SELECT
    'age_group'    AS age_group,
    'disease_name' AS disease_name,
    'astigmatic'   AS astigmatic,
    'tear_rate'    AS tear_rate,
    'lenses'       AS lenses;

-- Row 2: Column types
-- s = string/meta, d = discrete (categorical)
CREATE OR REPLACE VIEW orange_header_types AS
SELECT
    'd' AS age_group,
    'd' AS disease_name,
    'd' AS astigmatic,
    'd' AS tear_rate,
    'd' AS lenses;

-- Row 3: Roles
CREATE OR REPLACE VIEW orange_header_roles AS
SELECT
    ''     AS age_group,
    ''     AS disease_name,
    ''     AS astigmatic,
    ''     AS tear_rate,
    'class' AS lenses;

-- Data rows
CREATE OR REPLACE VIEW orange_data AS
SELECT
    p.age_group::TEXT AS age_group,
    d.disease_name::TEXT AS disease_name,
    CASE WHEN e.astigmatic THEN 'yes' ELSE 'no' END AS astigmatic,
    e.tear_rate::TEXT AS tear_rate,
    e.lenses::TEXT AS lenses
FROM EXAMINATION e
JOIN PATIENT p  ON e.patient_id = p.patient_id
JOIN DISEASE d  ON e.disease_id = d.disease_id;

-- Final export view
CREATE OR REPLACE VIEW orange_dataset AS
SELECT * FROM orange_header_names
UNION ALL
SELECT * FROM orange_header_types
UNION ALL
SELECT * FROM orange_header_roles
UNION ALL
SELECT * FROM orange_data;
