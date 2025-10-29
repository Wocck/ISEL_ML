-- ===========================================
-- POPULATE DATABASE
-- ===========================================

-- Insert doctors
INSERT INTO DOCTOR (name, specialization) VALUES
('Dr. Smith', 'Ophthalmologist'),
('Dr. Brown', 'Ophthalmologist'),
('Dr. Johnson', 'Optometrist');

-- Insert diseases
INSERT INTO DISEASE (disease_name) VALUES
('myope'),
('hypermetrope'),
('astigmatic');

-- Insert patients
INSERT INTO PATIENT (name, birth_date, age_group) VALUES
('Alice', '2000-05-10', 'young'),
('Bob', '1999-07-22', 'young'),
('Charlie', '1965-03-15', 'presbyopic'),
('Diana', '1970-08-30', 'presbyopic'),
('Ethan', '1985-11-04', 'pre-presbyopic'),
('Fiona', '1983-02-12', 'pre-presbyopic');

-- Insert examinations
INSERT INTO EXAMINATION (exam_date, astigmatic, tear_rate, lenses, patient_id, doctor_id, disease_id)
VALUES
-- young
('2024-01-10', TRUE, 'normal', 'hard', 1, 1, 1), -- young myope yes normal hard
('2024-01-11', FALSE, 'normal', 'soft', 2, 2, 1), -- young myope no normal soft
('2024-01-12', TRUE, 'reduced', 'none', 1, 1, 2), -- young hypermetrope yes reduced none
('2024-01-13', FALSE, 'normal', 'soft', 2, 1, 2), -- young hypermetrope no normal soft
('2024-01-14', FALSE, 'reduced', 'none', 2, 1, 2), -- young hypermetrope no reduced none

-- presbyopic
('2024-02-01', TRUE, 'reduced', 'none', 3, 3, 1), -- presbyopic myope yes reduced none
('2024-02-02', TRUE, 'normal', 'hard', 4, 2, 1), -- presbyopic myope yes normal hard
('2024-02-03', TRUE, 'reduced', 'none', 3, 2, 2), -- presbyopic hypermetrope yes reduced none
('2024-02-04', TRUE, 'normal', 'none', 4, 3, 2), -- presbyopic hypermetrope yes normal none
('2024-02-05', FALSE, 'normal', 'soft', 3, 3, 2), -- presbyopic hypermetrope no normal soft
('2024-02-06', FALSE, 'reduced', 'none', 4, 2, 2), -- presbyopic hypermetrope no reduced none

-- pre-presbyopic
('2024-03-01', TRUE, 'reduced', 'none', 5, 1, 1), -- pre-presbyopic myope yes reduced none
('2024-03-02', TRUE, 'normal', 'hard', 6, 2, 1), -- pre-presbyopic myope yes normal hard
('2024-03-03', FALSE, 'normal', 'soft', 5, 3, 1), -- pre-presbyopic myope no normal soft
('2024-03-04', TRUE, 'normal', 'none', 6, 1, 2), -- pre-presbyopic hypermetrope yes normal none
('2024-03-05', FALSE, 'normal', 'soft', 5, 1, 2); -- pre-presbyopic hypermetrope no normal soft
