# Domain Description – MedKnow Lenses System

The company MedKnow is a medical center that collects data about patients during eye examinations. Each patient is analyzed by a doctor, who checks vision problems such as myopia, hypermetropia, or astigmatism. The goal of the system is to help doctors choose the best type of contact lenses for each patient based on their eye condition and personal characteristics.

The dataset used by MedKnow includes information about the patient’s:
- age group (young, pre-presbyopic, or presbyopic), 
- the type of visual defect (myope or hypermetrope), 
- whether the patient has astigmatism,
- the tear production rate (normal or reduced). 
Based on these features, the system recommends the most suitable lens type: soft, hard, or none if lenses are not appropriate.

The system must support two perspectives. From the operational perspective, it should store data about patients, doctors, and eye examinations in a relational database. From the strategic perspective, it should use data-mining and machine learning techniques to identify useful patterns and provide decision support for lens prescription. For example, the system can learn simple classification rules (like the 1R or decision tree algorithms) that predict the best lens type for new patients based on previous data.