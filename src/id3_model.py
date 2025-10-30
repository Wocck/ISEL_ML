import pandas as pd
import math
from pathlib import Path
from typing import Optional, Any

class ID3Classifier:
    def __init__(self, dataset_file: Path):
        self.target_col = None
        self.tree = None
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
            raise ValueError(f"Column '{target_col}'does not exist in data.")
        return df

    def _entropy(self, series: pd.Series) -> float:
        # H(Y) = - sum p * log2 p
        counts = series.value_counts(normalize=True)
        ent = 0.0
        for p in counts:
            if p > 0:
                ent -= p * math.log2(p)
        return ent

    def _info_gain(self, df: pd.DataFrame, attr: str, target_col: str) -> float:
        # IG = H(Y) - sum_v (|Sv|/|S|) * H(Y|Sv)
        base_entropy = self._entropy(df[target_col])
        values = df[attr].unique()
        cond_entropy = 0.0

        for v in values:
            subset = df[df[attr] == v]
            weight = len(subset) / len(df)
            cond_entropy += weight * self._entropy(subset[target_col])

        gain = base_entropy - cond_entropy
        return gain

    def _majority_class(self, df: pd.DataFrame, target_col: str) -> str:
       return str(df[target_col].value_counts().idxmax())

    def _build_tree_recursive(self, df: pd.DataFrame, attrs: list, target_col: str):
        # case 1: all records are from the same class -> leaf
        unique_classes = df[target_col].unique()
        if len(unique_classes) == 1:
            return {
                "leaf": unique_classes[0]
            }

        # case 2: no atributes for partition -> leaf with majority class
        if len(attrs) == 0:
            return {
                "leaf": self._majority_class(df, target_col)
            }

        # choose atr with highest information gain
        best_attr = None
        best_gain = -1.0
        for a in attrs:
            gain = self._info_gain(df, a, target_col)
            if gain > best_gain:
                best_gain = gain
                best_attr = a

        # if gain is negative or zero then finish the tree
        if best_attr is None:
            return {
                "leaf": self._majority_class(df, target_col)
            }

        node = {
            "attr": best_attr,
            "branches": {}
        }

        # for each value of the selected attribute we build a subtree
        for v in df[best_attr].unique():
            subset = df[df[best_attr] == v]
            # delete atr, that we just used
            new_attrs = [x for x in attrs if x != best_attr]
            node["branches"][v] = self._build_tree_recursive(subset, new_attrs, target_col)

        return node

    def fit(self, target_col: str):
        if self.df is None:
            raise RuntimeError("Model has no data loaded. Use set_training_data() first!")
        
        self.df = self._normalize_df(self.df, target_col)
        self.target_col = target_col

        self.default_class = self._majority_class(self.df, target_col)
        attrs = [c for c in self.df.columns if c != target_col]

        self.tree = self._build_tree_recursive(self.df, attrs, target_col)
        self.fitted = True

    def _predict_row_from_tree(self, row: dict, node: dict) -> str:
        if "leaf" in node:
            return node["leaf"]

        attr = node["attr"]
        val = row.get(attr, None)
        if val is None:
            return str(self.default_class)

        if val in node["branches"]:
            return self._predict_row_from_tree(row, node["branches"][val])

        return str(self.default_class)

    def predict_from_values(self, age_group, disease_name, astigmatic, tear_rate):
        if not self.fitted or self.tree is None:
            raise RuntimeError("Model ID3 was not trained. Use fit() first!")

        row = {
            "age_group": str(age_group).strip(),
            "disease_name": str(disease_name).strip(),
            "astigmatic": "yes" if astigmatic else "no",
            "tear_rate": str(tear_rate).strip()
        }
        return self._predict_row_from_tree(row, self.tree)

    def predict_all_training(self):
        if self.df is None:
            raise RuntimeError("Model has no data loaded. Use set_training_data() first!")

        if not self.fitted or self.tree is None:
            raise RuntimeError("Model ID3 was not trained. Use fit() first!")
        
        preds = []
        for i in range(len(self.df)):
            row = {
                "age_group": str(self.df.iloc[i]["age_group"]).strip(),
                "disease_name": str(self.df.iloc[i]["disease_name"]).strip(),
                "astigmatic": str(self.df.iloc[i]["astigmatic"]).strip(),
                "tear_rate": str(self.df.iloc[i]["tear_rate"]).strip()
            }
            pred = self._predict_row_from_tree(row, self.tree)
            preds.append(pred)
        return preds

    def score(self):
        if self.df is None:
            raise RuntimeError("Model has no data loaded. Use set_training_data() first!")
        if not self.fitted:
            raise RuntimeError("Model ID3 was not trained. Use fit() first!")

        preds = self.predict_all_training()
        real = self.df[self.target_col].tolist()

        total = len(real)
        if total == 0:
            return 0.0

        ok = 0
        for p, r in zip(preds, real):
            if p == r:
                ok += 1
        return ok / total

    def print_tree(self, node: Optional[dict[str, Any]] = None, indent: str = "") -> None:
        if node is None:
            node = self.tree
            if node is None:
                print("Model ID3 was not trained - Tree is empty. Use fit() first!")
                return

        if isinstance(node, dict) and "leaf" in node:
            print(indent + "->", node["leaf"])
            return

        if isinstance(node, dict):
            attr = node.get("attr", "?")
            for val, subtree in node.get("branches", {}).items():
                print(indent + f"[{attr} == {val}]")
                self.print_tree(subtree, indent + "  ")
