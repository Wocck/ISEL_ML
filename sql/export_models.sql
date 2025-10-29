-- ==================================================
-- TAB EXPORT VIEW
-- ==================================================

CREATE OR REPLACE VIEW models_dataset AS
SELECT
    p.age_group::TEXT AS age_group,
    d.disease_name::TEXT AS disease_name,
    CASE 
        WHEN e.astigmatic THEN 'yes'
        ELSE 'no'
    END AS astigmatic,
    e.tear_rate::TEXT AS tear_rate,
    e.lenses::TEXT AS lenses
FROM EXAMINATION e
JOIN PATIENT p  ON e.patient_id = p.patient_id
JOIN DISEASE d  ON e.disease_id = d.disease_id;