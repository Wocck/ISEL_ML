import pandas as pd
import math
from pathlib import Path

class NaiveBayesClassifier:
    def __init__(self, dataset_file: Path):
        self.target_col = None
        self.class_priors = {}
        self.cond_probs = {}
        self.default_class = None
        self.fitted = False
        self.df = None

    def set_training_data(self, df: pd.DataFrame):
        self.df = df.copy()

    @staticmethod
    def _normalize_df(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        # norm everything as string
        df = df.copy()
        for c in df.columns:
            df[c] = df[c].astype(str).str.strip()
        if target_col not in df.columns:
            raise ValueError(f"Column of class '{target_col}' does not exist in data.")
        return df

    def fit(self, target_col: str):
        if self.df is None:
            raise RuntimeError("Model has no data loaded. Use set_training_data() first!")
        
        self.df = self._normalize_df(self.df, target_col)
        self.target_col = target_col

        # Default class = most common label
        self.default_class = self.df[target_col].value_counts().idxmax()

        classes = self.df[target_col].unique().tolist()
        attrs = [c for c in self.df.columns if c != target_col]

        # prior P(class)
        total = len(self.df)
        for c in classes:
            count_c = len(self.df[self.df[target_col] == c])
            self.class_priors[c] = count_c / total

        # conditional P(attr=value | class) - Laplace smoothing
        # structure: cond_probs[attr][value][class] = probability
        self.cond_probs = {attr: {} for attr in attrs}

        for attr in attrs:
            # all values for given attribute
            all_values = self.df[attr].unique().tolist()

            for v in all_values:
                if v not in self.cond_probs[attr]:
                    self.cond_probs[attr][v] = {}

                for c in classes:
                    subset_c = self.df[self.df[target_col] == c]
                    count_v_c = len(subset_c[subset_c[attr] == v])

                    # Laplace smoothing:
                    # (count_v_c + 1) / (|class_c| + |values_of_attr|)
                    prob = (count_v_c + 1) / (len(subset_c) + len(all_values))
                    self.cond_probs[attr][v][c] = prob

            # If a new (unseen) attribute value appears during prediction,
            # assign it a small fallback probability to avoid zero-probability errors.
            # This fallback is stored under a special key "__UNK__".
            unk = {}
            for c in classes:
                subset_c = self.df[self.df[target_col] == c]
                unk[c] = 1 / (len(subset_c) + len(all_values))
            self.cond_probs[attr]["__UNK__"] = unk

        self.fitted = True

    def _predict_single_rowdict(self, row: dict) -> str:
        # Calculate log(P(class)) + sum(log(P(attr=value|class)))
        if not self.fitted:
            raise RuntimeError("Model Naive Bayes was not trained. Use fit() first!")

        best_class = None
        best_score = None

        for c in self.class_priors.keys():
            score = math.log(self.class_priors[c] + 1e-12)

            for attr, val in row.items():
                if attr == self.target_col:
                    continue
                
                if val in self.cond_probs[attr]:
                    prob_val = self.cond_probs[attr][val][c]
                else:
                    prob_val = self.cond_probs[attr]["__UNK__"][c]

                score += math.log(prob_val + 1e-12)

            if best_score is None or score > best_score:
                best_score = score
                best_class = c

        return str(best_class or self.default_class)

    def predict_from_values(self, age_group, disease_name, astigmatic, tear_rate):
        row = {
            "age_group": str(age_group).strip(),
            "disease_name": str(disease_name).strip(),
            "astigmatic": "yes" if astigmatic else "no",
            "tear_rate": str(tear_rate).strip()
        }
        return self._predict_single_rowdict(row)

    def score(self):
        if self.df is None:
            raise RuntimeError("Model has no data loaded. Use set_training_data() first!")
        if not self.fitted:
            raise RuntimeError("Model Naive Bayes was not trained. Use fit() first!")
        if self.target_col is None:
            raise ValueError("No target column found. Use fit() first!")
        
        correct = 0
        total = len(self.df)

        for i in range(total):
            row = {
                "age_group": str(self.df.iloc[i]["age_group"]).strip(),
                "disease_name": str(self.df.iloc[i]["disease_name"]).strip(),
                "astigmatic": str(self.df.iloc[i]["astigmatic"]).strip(),
                "tear_rate": str(self.df.iloc[i]["tear_rate"]).strip()
            }
            pred = self._predict_single_rowdict(row)
            real = str(self.df.iloc[i][self.target_col]).strip()
            if pred == real:
                correct += 1

        if total == 0:
            return 0.0
        return correct / total
