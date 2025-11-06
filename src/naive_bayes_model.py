import pandas as pd

from evaluation import evaluate_on_test
class NaiveBayesClassifier:
    def __init__(self):
        self.target_col = None
        self.class_priors = {}
        self.cond_probs = {}
        self.default_class = None
        self.fitted = False
        self.df = None

    def set_training_data(self, df: pd.DataFrame):
        # simple normalization
        self.df = df.astype(str).apply(lambda col: col.str.strip())

    def fit(self, target_col: str):
        if self.df is None:
            raise RuntimeError("Model has no data loaded. Use set_training_data() first!")
        
        self.target_col = target_col

        # Default class = most common label
        self.default_class = self.df[target_col].value_counts().idxmax()

        classes = self.df[target_col].unique().tolist()
        attrs = [c for c in self.df.columns if c != target_col]

        # Prior probability P(class) estimated as relative frequency in training data
        total = len(self.df)
        for c in classes:
            count_c = len(self.df[self.df[target_col] == c])
            self.class_priors[c] = count_c / total

        # Conditional probability P(attribute = value | class)
        # Estimated with Laplace smoothing to avoid zero probabilities:
        # (count + 1) / (class_count + number_of_possible_values)
        self.cond_probs = {attr: {} for attr in attrs}

        for attr in attrs:
            # All values for given attribute
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
        # Compute P(class) * Π P(attr=value | class)
        if not self.fitted:
            raise RuntimeError("Model Naive Bayes was not trained. Use fit() first!")

        best_class = None
        best_score = None

        for c in self.class_priors:
            score = self.class_priors[c]

            for attr, val in row.items():
                if attr == self.target_col:
                    continue
                # Use conditional probability if known, otherwise fallback
                if val in self.cond_probs[attr]:
                    prob_val = self.cond_probs[attr][val][c]
                else:
                    prob_val = self.cond_probs[attr]["__UNK__"][c]
                # Multiply probabilities
                score *= prob_val
            # Keep the class with the highest probability score
            if best_score is None or score > best_score:
                best_score = score
                best_class = c

        return str(best_class or self.default_class)

    def predict_row(self, row):
        if not self.fitted:
            raise RuntimeError("Model Naive Bayes was not trained. Use fit() first!")
        # Normalize input row to match training data formatting
        rowdict = {k: str(v).strip() for k, v in row.items()}
        return self._predict_single_rowdict(rowdict)


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
            row = {k: str(self.df.iloc[i][k]).strip() for k in self.df.columns if k != self.target_col}
            pred = self._predict_single_rowdict(row)
            real = str(self.df.iloc[i][self.target_col]).strip()
            if pred == real:
                correct += 1

        if total == 0:
            return 0.0
        return correct / total

    def run_evaluate(self, test_df):
        print("\n===============================")
        print(" Model: Naive Bayes")
        print("===============================")
        print("Accuracy:")
        print(f"  Train: {self.score()*100:.2f}%")
        print(f"  Test:  {evaluate_on_test(self, test_df, 'lenses')*100:.2f}%")
