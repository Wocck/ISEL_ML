import pandas as pd
from tqdm import tqdm

from evaluation import evaluate_on_test

class OneRClassifier:
    def __init__(self):
        self.best_attribute = None
        self.rules = {}
        self.default_class = None
        self.target_col = None
        self.fitted = False
        self.df = None

    def set_training_data(self, df: pd.DataFrame):
        self.df = df.copy()

    @staticmethod
    def _normalize_df(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        # norm everything as string
        df = df.copy()
        for c in df.columns:
            if df[c].dtype == bool:
                df[c] = df[c].map({True: "yes", False: "no"})
        for c in df.columns:
            df[c] = df[c].astype(str).str.strip()
        if target_col not in df.columns:
            raise ValueError(f"Brak kolumny docelowej '{target_col}' w danych.")
        return df

    def _build_rules_for_attribute(self, attr, target_col):
        if self.df is None:
            raise RuntimeError("Model has no data loaded. Use set_training_data() first!")
        counts = (
            self.df.groupby([attr, target_col])
              .size()
              .reset_index(name='count')
        )
        rules = {}
        for value in counts[attr].unique():
            subset = counts[counts[attr] == value]
            majority_row = subset.loc[subset['count'].idxmax()]
            rules[value] = majority_row[target_col]

        predictions = self.df[attr].map(rules)
        accuracy = (predictions == self.df[target_col]).mean()
        return rules, accuracy

    def fit(self, target_col):
        if self.df is None:
            raise RuntimeError("Model has no data loaded. Use set_training_data() first!")
        self.df = self._normalize_df(self.df, target_col)
        self.target_col = target_col

        candidate_attributes = [c for c in self.df.columns if c != target_col]
        best_attr_local = None
        best_rules_local = None
        best_acc_local = -1.0

        for attr in tqdm(candidate_attributes, desc="Training atr", unit="atr"):
            rules_attr, acc_attr = self._build_rules_for_attribute(attr, target_col)
            if acc_attr > best_acc_local:
                best_acc_local = acc_attr
                best_attr_local = attr
                best_rules_local = rules_attr

        self.best_attribute = best_attr_local
        self.rules = best_rules_local
        self.default_class = self.df[target_col].value_counts().idxmax()
        self.fitted = True

    def predict_row(self, row):
        if not self.fitted:
            raise RuntimeError("Model was not trained. Use fit() first.")
        attr_value = str(row[self.best_attribute]).strip()
        if self.rules:
            return self.rules.get(attr_value, self.default_class)
        else:
            print("[Error] This should not happen!!!")

    def predict(self):
        if self.df is None:
            raise RuntimeError("Model has no data loaded. Use set_training_data() first!")
        df = self.df.copy()
        for c in df.columns:
            df[c] = df[c].astype(str).str.strip()
        return [self.predict_row(df.iloc[i]) for i in range(len(df))]

    def score(self):
        if self.df is None:
            raise RuntimeError("Model has no data loaded. Use set_training_data() first!")
        df = self.df.copy()
        for c in df.columns:
            df[c] = df[c].astype(str).str.strip()
        preds = self.predict()
        real = df[self.target_col].tolist()
        total = len(real)
        if total == 0:
            return 0.0
        correct = sum(1 for p, r in zip(preds, real) if p == r)
        return correct / total

    def pretty_print_rules(self):
        if not self.fitted:
            print("Model not trained.")
            return
        print("1R model on atr:", self.best_attribute)
        if self.rules:
            for attr_val, cls in self.rules.items():
                print(f"  IF {self.best_attribute} == {attr_val} THEN {self.target_col} = {cls}")
        print(f"  ELSE {self.target_col} = {self.default_class}  (default)")
        print()
        
    def run_evaluate(self, test_df):
        print("\n===============================")
        print(" Model: One Rule (1R)")
        print("===============================")
        print(f"Selected attribute: {self.best_attribute}")
        print("Rules:")
        if self.rules is not None:
            for val, cls in self.rules.items():
                print(f"  IF {self.best_attribute} = {val} → lenses = {cls}")
        print(f"Default class: {self.default_class}\n")

        print("Accuracy:")
        print(f"  Train: {self.score()*100:.2f}%")
        print(f"  Test:  {evaluate_on_test(self, test_df, 'lenses')*100:.2f}%")
