import pandas as pd
import math
from typing import Dict, Optional
from dataclasses import dataclass


# Tree
@dataclass
class TreeNode:
    attribute: Optional[str] = None       # attr name if node
    branches: Optional[Dict[str, "TreeNode"]] = None
    label: Optional[str] = None           # if leaf

    def is_leaf(self) -> bool:
        return self.label is not None


class ID3Classifier:
    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.target_col: Optional[str] = None
        self.tree: Optional[TreeNode] = None
        self.default_class: Optional[str] = None
        self.fitted: bool = False

    def set_training_data(self, df: pd.DataFrame):
        # simple normalization
        self.df = df.astype(str).apply(lambda col: col.str.strip())

    def _entropy(self, col: pd.Series) -> float:
        counts = col.value_counts()
        total = len(col)
        return -sum((count / total) * math.log2(count / total) for count in counts if count > 0)

    def _info_gain(self, df: pd.DataFrame, attr: str) -> float:
        base_entropy = self._entropy(df[self.target_col])
        values = df[attr].unique()

        entropy_after_split = 0.0
        for v in values:
            subset = df[df[attr] == v]
            weight = len(subset) / len(df)
            entropy_after_split += weight * self._entropy(subset[self.target_col])

        return base_entropy - entropy_after_split

    def _majority_class(self, df: pd.DataFrame) -> str:
        counts = df[self.target_col].value_counts()
        majority_class = counts.index[0]
        return str(majority_class)

    def _build_tree(self, df: pd.DataFrame, attrs) -> TreeNode:
        classes = df[self.target_col].unique()

        # if only one class -> leaf
        if len(classes) == 1:
            return TreeNode(label=str(classes[0]))

        # if no attr -> leaf with majority class
        if not attrs:
            return TreeNode(label=self._majority_class(df))

        gains = {attr: self._info_gain(df, attr) for attr in attrs}
        best_attr = max(gains, key=lambda a: gains[a])
        branches = {}
        for value in df[best_attr].unique():
            # divide dataset by best_attr
            subset = df[df[best_attr] == value]
            if subset.empty:
                # if there is no examples -> leaf with majority class
                branches[value] = TreeNode(label=self._majority_class(df))
            else:
                # continue with dividing dataset by remaining attributes
                remaining_attrs = [a for a in attrs if a != best_attr]
                branches[value] = self._build_tree(subset, remaining_attrs)

        # return ID3 Tree
        return TreeNode(attribute=best_attr, branches=branches)


    def fit(self, target_col: str):
        if self.df is None:
            raise RuntimeError("Use set_training_data() first.")

        self.target_col = target_col
        # Default class is the global majority, used as fallback in prediction
        self.default_class = self._majority_class(self.df)
        attrs = [c for c in self.df.columns if c != target_col]

        self.tree = self._build_tree(self.df, attrs)
        self.fitted = True

    def _predict_from_tree(self, node: TreeNode, row: Dict[str, str]) -> str:
        # If node is a leaf -> return its class label
        if node.is_leaf():
            assert node.label is not None
            return node.label

        # Otherwise follow the branch matching the value of the node's attribute
        assert node.attribute is not None
        assert node.branches is not None

        val = row.get(node.attribute)
        # If the value exists among the branches, continue recursively
        if val in node.branches:
            return self._predict_from_tree(node.branches[val], row)

        # If unseen attribute value -> fallback to default class
        assert self.default_class is not None
        return self.default_class

    def predict_row(self, row):
        if not self.fitted or self.tree is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        row = {k: str(v).strip() for k, v in row.items()}
        return self._predict_from_tree(self.tree, row)

    def predict_all_training(self):
        assert self.df is not None
        return [self.predict_row(self.df.iloc[i].to_dict()) for i in range(len(self.df))]

    def score(self):
        assert self.df is not None
        # Compare predicted class vs. actual class
        preds = self.predict_all_training()
        real = self.df[self.target_col].tolist()
        # Accuracy = correct predictions / total samples
        return sum(p == r for p, r in zip(preds, real)) / len(real)

    def print_tree(self, node: Optional[TreeNode] = None, indent=""):
        if node is None:
            node = self.tree

        if node is None:
            print("Empty tree.")
            return

        if node.is_leaf():
            print(indent + "->", node.label)
            return

        if node.branches:
            for val, child in node.branches.items():
                print(f"{indent}[{node.attribute} = {val}]")
                self.print_tree(child, indent + "  ")

    def run_evaluate(self, test_df):
        print("\n===============================")
        print(" Model: ID3 Decision Tree")
        print("===============================")
        print("Tree structure:")
        self.print_tree()

        print("\nAccuracy:")
        print(f"  Train: {self.score()*100:.2f}%")

        from src.evaluation import evaluate_on_test
        print(f"  Test:  {evaluate_on_test(self, test_df, self.target_col)*100:.2f}%")
